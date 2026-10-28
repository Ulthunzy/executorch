import org.json.JSONObject;
import org.pytorch.executorch.extension.llm.Qwen35MoeModule;

/** Focused SM8850 text, history, reset and close check with the current libraries. */
public final class Qwen35MoeSm8850SmokeTest {
  private static JSONObject run(
      Qwen35MoeModule model, String label, String prompt, int limit) throws Exception {
    StringBuilder answer = new StringBuilder();
    String stats = model.generate(prompt, limit, 0, false, answer::append);
    JSONObject result = new JSONObject(stats);
    if (answer.length() == 0 || answer.indexOf("\ufffd") >= 0
        || result.getInt("context_capacity") != 2048
        || result.getInt("generated_tokens") < 1
        || result.getInt("generated_tokens") > limit) {
      throw new AssertionError(label + ": invalid JNI output: " + stats);
    }
    result.put("label", label);
    result.put("answer", answer.toString());
    System.out.println("JNI_TURN " + result.toString());
    return result;
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 3 && args.length != 5)
      throw new IllegalArgumentException("model.pte workspace libraries OR model shared tokenizer workspace libraries");
    Qwen35MoeModule.Config config = args.length == 3
        ? new Qwen35MoeModule.Config(args[0], args[1], args[2])
        : new Qwen35MoeModule.Config(args[0], args[1], args[2], args[3], args[4]);
    config.contextLength = 2048;
    config.expertsPerLayer = 64;
    config.enableVision = false;
    Qwen35MoeModule model = new Qwen35MoeModule(config);
    try {
      JSONObject first = run(model, "first", "你好，请用一句话介绍一下你自己。", 48);
      JSONObject followup = run(model, "followup", "我刚才问了你什么？请简短回答。", 32);
      if (followup.getInt("context_tokens") <= first.getInt("context_tokens"))
        throw new AssertionError("Conversation context did not advance");
      model.resetContext();
      JSONObject reset = run(model, "reset", "1加1等于几？只回答数字。", 8);
      if (reset.getInt("context_tokens") >= followup.getInt("context_tokens"))
        throw new AssertionError("resetContext did not clear context");
    } finally {
      model.close();
    }
    model.close();
    try {
      model.generate("你好", 1, 0, false, piece -> {});
      throw new AssertionError("Closed module accepted generate");
    } catch (IllegalStateException expected) {
    }
    System.out.println("SM8850_JNI_TEXT_HISTORY_RESET_CLOSE_OK");
  }
}
