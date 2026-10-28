from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def compile_loader(tmp_path, *, old_size_guard=False):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "runner/decode/runner.cpp").read_text()
    start = source.index("      if (kind == 0) {")
    end, depth = start, 0
    for end in range(start, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if source[end] == "}" and depth == 0:
            break
    block = source[start : end + 1]
    if old_size_guard:
        block = block.replace(
            "if (kind == 0) {",
            "if (kind == 0) {\n if (size != 1296) return Error::DelegateInvalidCompatibility;",
            1,
        )
    harness = r"""
#include "runner/cache_encoding.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <string>
enum class Error { DelegateInvalidCompatibility };
using DelegateHandle = void;
struct Stage { unsigned kind, layer; };
struct Result {
  bool ok;
  Result(Error) : ok(false) {}
  Result(DelegateHandle* p) : ok(true) { delete static_cast<Stage*>(p); }
};
std::string model_root;
std::vector<uint8_t> disk, cache_encoding;
std::vector<uint8_t> bytes(const std::string&) { return disk; }
Result load(const std::vector<uint8_t>& payload, size_t aux_size=0) {
  const unsigned kind=0,layer=0;
  const size_t size=payload.size();
  std::vector<uint8_t> wrapped(128+size);
  std::copy(payload.begin(),payload.end(),wrapped.begin()+128);
  const uint8_t* p=wrapped.data();
@BLOCK@
  return Error::DelegateInvalidCompatibility;
}
std::vector<uint8_t> metadata(unsigned version) {
  const size_t stride=version==1?16:24;
  std::vector<uint8_t> data(16+80*stride);
  std::memcpy(data.data(),"Q36CACHE",8);
  uint32_t header[]={version,40};std::memcpy(data.data()+8,header,8);
  for(size_t i=0;i<80;++i) {
    const bool half=version==2 && i%2==1 && i/2%4!=3;
    float scale=half?0:0.25f;int32_t offset=half?0:-100;
    uint64_t elements=512;uint32_t dtype=half?1:0;
    auto* e=data.data()+16+i*stride;
    std::memcpy(e,&scale,4);std::memcpy(e+4,&offset,4);std::memcpy(e+8,&elements,8);
    if(version==2)std::memcpy(e+16,&dtype,4);
  }
  return data;
}
int main(int argc,char**argv) {
  const std::string test=argv[1];
  disk=metadata(test=="legacy"?1:2);
  auto payload=disk;size_t aux=0;
  if(test=="truncated")payload.pop_back();
  if(test=="mismatch")payload[16]^=1;
  if(test=="bad_header"){disk[12]=39;payload=disk;}
  if(test=="bad_dtype"){disk[16+24+16]=2;payload=disk;}
  if(test=="bad_role"){
    std::copy_n(disk.data()+16+24,24,disk.data()+16);payload=disk;
  }
  if(test=="aux")aux=1;
  const bool expected=test=="legacy"||test=="typed";
  const bool actual=load(payload,aux).ok;
  if(actual!=expected)return 1;
  if(actual && cache_encoding!=disk)return 2;
}
""".replace(
        "@BLOCK@", block
    )
    path = tmp_path / ("old.cpp" if old_size_guard else "new.cpp")
    path.write_text(harness)
    binary = path.with_suffix("")
    subprocess.run(
        [compiler, "-std=c++17", "-O1", "-I", str(ROOT), str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    return compile_loader(tmp_path_factory.mktemp("decode-cache-blob"))


@pytest.mark.parametrize(
    "case",
    [
        "legacy",
        "typed",
        "truncated",
        "mismatch",
        "bad_header",
        "bad_dtype",
        "bad_role",
        "aux",
    ],
)
def test_actual_decode_initializer_cache_blob(loader, case):
    subprocess.run([str(loader), case], check=True, capture_output=True, text=True)


def test_old_1296_guard_would_reject_typed_blob(tmp_path):
    loader = compile_loader(tmp_path, old_size_guard=True)
    assert subprocess.run([str(loader), "typed"], capture_output=True).returncode != 0
