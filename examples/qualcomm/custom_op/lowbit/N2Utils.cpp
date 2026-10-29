#include "../htp_arch.h"
#include "N2Utils.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#define vmem(p)                    \
  (*reinterpret_cast<HVX_Vector*>( \
      const_cast<void*>(static_cast<const void*>(p))))

void n2_expand_in_place(uint8_t* buffer, int compressed_bytes) {
  // 输入长度须为 128 B 的倍数，buffer 容量至少为两倍长度；调用者提供 VTCM
  // 对齐区。
  alignas(128) static const uint8_t lut[128] = {
      153, 0, 158, 0, 146, 0, 151, 0, 233, 0, 238, 0, 226, 0, 231, 0,
      41,  0, 46,  0, 34,  0, 39,  0, 121, 0, 126, 0, 114, 0, 119, 0,
      0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0,
      0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0,
      0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0,
      0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0,
      0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0,
      0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0};
  const auto table = vmem(lut);
  const auto mask = Q6_Vb_vsplat_R(15);
#if QCOM_HTP_V81
  int offset = compressed_bytes - 128;
  for (; offset >= 384; offset -= 512) {
    const auto input0 = vmem(buffer + offset - 0);
    const auto input1 = vmem(buffer + offset - 128);
    const auto input2 = vmem(buffer + offset - 256);
    const auto input3 = vmem(buffer + offset - 384);
    const auto low0 = Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(input0, mask), table, 0);
    const auto high0 = Q6_Vb_vlut32_VbVbI(Q6_Vub_vlsr_VubR(input0, 4), table, 0);
    const auto output0 = Q6_W_vshuff_VVR(high0, low0, -1);
    vmem(buffer + (offset - 0) * 2) = Q6_V_lo_W(output0);
    vmem(buffer + (offset - 0) * 2 + 128) = Q6_V_hi_W(output0);
    const auto low1 = Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(input1, mask), table, 0);
    const auto high1 = Q6_Vb_vlut32_VbVbI(Q6_Vub_vlsr_VubR(input1, 4), table, 0);
    const auto output1 = Q6_W_vshuff_VVR(high1, low1, -1);
    vmem(buffer + (offset - 128) * 2) = Q6_V_lo_W(output1);
    vmem(buffer + (offset - 128) * 2 + 128) = Q6_V_hi_W(output1);
    const auto low2 = Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(input2, mask), table, 0);
    const auto high2 = Q6_Vb_vlut32_VbVbI(Q6_Vub_vlsr_VubR(input2, 4), table, 0);
    const auto output2 = Q6_W_vshuff_VVR(high2, low2, -1);
    vmem(buffer + (offset - 256) * 2) = Q6_V_lo_W(output2);
    vmem(buffer + (offset - 256) * 2 + 128) = Q6_V_hi_W(output2);
    const auto low3 = Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(input3, mask), table, 0);
    const auto high3 = Q6_Vb_vlut32_VbVbI(Q6_Vub_vlsr_VubR(input3, 4), table, 0);
    const auto output3 = Q6_W_vshuff_VVR(high3, low3, -1);
    vmem(buffer + (offset - 384) * 2) = Q6_V_lo_W(output3);
    vmem(buffer + (offset - 384) * 2 + 128) = Q6_V_hi_W(output3);
  }
  for (; offset >= 0; offset -= 128) {
    const auto input = vmem(buffer + offset);
    const auto low = Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(input, mask), table, 0);
    const auto high = Q6_Vb_vlut32_VbVbI(Q6_Vub_vlsr_VubR(input, 4), table, 0);
    const auto output = Q6_W_vshuff_VVR(high, low, -1);
    vmem(buffer + offset * 2) = Q6_V_lo_W(output);
    vmem(buffer + offset * 2 + 128) = Q6_V_hi_W(output);
  }
#else
  // Expand backwards because source and destination share a VTCM tile.
  for (int offset = compressed_bytes - 128; offset >= 0; offset -= 128) {
    const auto input = vmem(buffer + offset);
    const auto low = Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(input, mask), table, 0);
    const auto high = Q6_Vb_vlut32_VbVbI(Q6_Vub_vlsr_VubR(input, 4), table, 0);
    const auto output = Q6_W_vshuff_VVR(high, low, -1);
    vmem(buffer + offset * 2) = Q6_V_lo_W(output);
    vmem(buffer + offset * 2 + 128) = Q6_V_hi_W(output);
  }
#endif
}
