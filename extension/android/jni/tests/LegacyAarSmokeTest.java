import java.util.Arrays;
import java.util.Collections;
import org.json.JSONObject;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.textencoder.TextEncoderModule;

/** Compile against the OLD AAR, then run unchanged bytecode with the NEW AAR. */
public final class LegacyAarSmokeTest {
  private static final String SYSTEM = "You are a helpful assistant.";
  private static final String QUESTION = "你好，你是谁？";

  private static final class Output implements LlmCallback {
    final StringBuilder text = new StringBuilder();
    String stats;
    LlmModule stopModule;

    @Override
    public void onResult(String token) {
      text.append(token);
      if (stopModule != null) {
        stopModule.stop();
        stopModule = null;
      }
    }

    @Override
    public void onStats(String value) {
      stats = value;
    }
  }

  private static void check(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }

  public static void main(String[] args) throws Exception {
    check(args.length == 4, "KV_PTE TOKENIZER ENCODER_PTE HYBRID_PTE");
    LlmModule module = new LlmModule(4, args[0], args[1], SYSTEM, "lincal",
        0.0f, Collections.emptyList());
    check(module.load() == 0, "legacy int load failed");
    Output first = new Output();
    check(module.generateWithChat(QUESTION, 16, first, true) == 0, "legacy chat failed");
    String prefix = "<|im_start|>system\n" + SYSTEM +
        "<|im_end|>\n<|im_start|>user\n" + QUESTION +
        "<|im_end|>\n<|im_start|>assistant\n";
    check(first.text.toString().startsWith(prefix), "wrong legacy chat template");
    check(first.text.substring(prefix.length()).startsWith("<think>"),
        "legacy chat should leave thinking enabled");
    check(new JSONObject(first.stats).getInt("generated_tokens") == 16,
        "chat seqLen must be a new-token budget");
    Output second = new Output();
    check(module.generateWithChat("1+1是多少？", 1, second, true) == 0, "second chat failed");
    check(!second.text.toString().contains("<|im_start|>system"), "system repeated");
    check(second.text.toString().startsWith("\n<|im_start|>user\n1+1是多少？"),
        "second turn must match the evaluation driver's separator");
    module.unload();
    module.unload();
    module.stop();
    check(module.load() == 0, "load after unload failed");
    Output reloaded = new Output();
    check(module.generateWithChat(QUESTION, 16, reloaded, true) == 0, "reload chat failed");
    check(first.text.toString().equals(reloaded.text.toString()), "unload retained KV");
    module.resetContext();
    Output noEcho = new Output();
    check(module.generateWithChat(QUESTION, 16, noEcho, false) == 0, "echo=false failed");
    check(first.text.substring(prefix.length()).equals(noEcho.text.toString()),
        "echo changed generation or thinking");
    module.resetContext();
    Output cancelled = new Output();
    cancelled.stopModule = module;
    check(module.generateWithChat(QUESTION, 16, cancelled, true) == 0, "cancel failed");
    check(new JSONObject(cancelled.stats).getInt("prompt_tokens") == 0,
        "echo cancellation should precede first AR");
    Output resumed = new Output();
    check(module.generateWithChat(QUESTION, 16, resumed, true) == 0, "resume failed");
    check(first.text.toString().equals(resumed.text.toString()), "cancel lost system prompt");
    module.unload();
    check(module.generate("hello", 0, new Output(), false, 0, 0, 0) == 0x12,
        "legacy generate must return native error status");
    module.resetNative();

    LlmModule explicit = new LlmModule(4, args[0], args[1], SYSTEM, "lincal",
        0.0f, Collections.emptyList(), 0, 0);
    check(explicit.generateWithChat(QUESTION, 1, new Output(), false) == 0,
        "legacy 9-argument constructor/lazy load failed");
    explicit.resetNative();

    LlmModule hybrid = new LlmModule(4, args[3], args[1], "", "qwen3",
        0.0f, Collections.emptyList());
    check(hybrid.load() == 0, "legacy hybrid auto-detection failed");
    hybrid.resetNative();

    TextEncoderModule encoder = new TextEncoderModule(args[2], 4, 4);
    check(Arrays.equals(encoder.encode(new int[] {2, 3, 4, 5},
        new float[] {1, 0, 1, 0}), new float[] {2, 0, 4, 0}), "encoder mismatch");
    float[] text = encoder.encodeText(args[1], "你好");
    check(text.length == 4 && text[0] > 0 && text[3] == 0,
        "encoder tokenize/pad mismatch");
    encoder.resetNative();
    Module portable = Module.load(args[2]);
    check(portable.loadMethod("forward") == 0, "int loadMethod failed");
    check(portable.loadMethod("missing") != 0, "loadMethod failure status missing");
    portable.destroy();
    System.out.println("PASS: unchanged legacy bytecode, chat/echo/think, unload/reload, " +
        "KV/hybrid detection, cancellation, TextEncoder, Module.loadMethod");
    System.exit(0);
  }
}
