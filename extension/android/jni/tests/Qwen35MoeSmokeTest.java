import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicInteger;
import org.pytorch.executorch.extension.llm.Qwen35MoeModule;

/** Device test with a complete PTE or an extracted real MoE deployment. */
public final class Qwen35MoeSmokeTest {
  private static void check(boolean value, String message) {
    if (!value)
      throw new AssertionError(message);
  }
  private static final class Output implements Qwen35MoeModule.Callback {
    final StringBuilder text = new StringBuilder();
    @Override
    public void onResult(String piece) {
      text.append(piece);
      System.out.print(piece);
      System.out.flush();
    }
  }
  public static void main(String[] args) throws Exception {
    // Exercise BPE byte boundaries without depending on the generated answer.
    Output unicode = new Output();
    Class<?> stream = Class.forName("org.pytorch.executorch.extension.llm.Qwen35MoeModule$Stream");
    Constructor<?> ctor = stream.getDeclaredConstructors()[0];
    ctor.setAccessible(true);
    Object decoder = ctor.newInstance(unicode);
    Method bytes = stream.getDeclaredMethod("onBytes", byte[].class);
    bytes.setAccessible(true);
    Method finish = stream.getDeclaredMethod("finish");
    finish.setAccessible(true);
    for (byte b : "中文😀".getBytes(StandardCharsets.UTF_8)) bytes.invoke(decoder, new byte[] {b});
    finish.invoke(decoder);
    check(unicode.text.toString().equals("中文😀"), "UTF-8 token boundaries");
    System.out.println("\nUNICODE_OK");

    Qwen35MoeModule.Config config = args.length == 4
        ? new Qwen35MoeModule.Config(args[0], args[1], args[2])
        : new Qwen35MoeModule.Config(args[0], args[1], args[2], args[3], args[4]);
    String imagePath = args[args.length == 4 ? 3 : 5];
    config.enableVision = true;
    try (Qwen35MoeModule module = new Qwen35MoeModule(config)) {
      module.resetContext();
      try {
        new Qwen35MoeModule(config);
        throw new AssertionError("Second live module accepted");
      } catch (IllegalStateException expected) {
      }
      Output first = new Output();
      AtomicInteger callbacks = new AtomicInteger();
      String stats = module.generate("请用一句话介绍你自己。", 24, 0, false, piece -> {
        first.onResult(piece);
        if (callbacks.incrementAndGet() == 2) {
          try {
            Thread.sleep(12000);
          } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(error);
          }
        }
      });
      System.out.println("\nTEXT_STATS " + stats);
      check(first.text.length() > 0, "No text tokens");
      check(first.text.indexOf("�") < 0, "Broken UTF-8");
      check(first.text.indexOf("<think>") < 0, "Thinking disabled was ignored");
      check(stats.contains("\"kv_cache_tokens\":"), "Missing cumulative KV length");
      check(stats.contains("\"context_capacity\":"), "Missing context capacity");
      // Human/UI think time must not leave a live FastRPC invocation behind.
      Thread.sleep(12000);
      Output follow = new Output();
      System.out.println(
          "FOLLOW_STATS " + module.generate("你能帮我做什么？", 8, 0, false, follow));
      module.resetContext();
      Output thinking = new Output();
      String thinkingStats =
          module.generate("请简短思考，再回答：1+1等于多少？", 512, 0, true, thinking);
      System.out.println("\nTHINKING_STATS " + thinkingStats);
      check(thinking.text.toString().startsWith("<think>\n"), "Missing thinking prefix");
      check(thinking.text.length() > "<think>\n".length(), "No generated thinking-mode text");
      check(thinking.text.indexOf("</think>") > 0, "Thinking boundary was filtered");
      check(thinkingStats.contains("\"enable_thinking\":true"), "Missing thinking mode");
      module.resetContext();
      try {
        module.generateWithImage("/does/not/exist.jpg", "描述图片", 8, 0, false, new Output());
        throw new AssertionError("Bad image accepted");
      } catch (IllegalStateException expected) {
        System.out.println("BAD_IMAGE_OK");
      }
      Output image = new Output();
      String imageStats =
          module.generateWithImage(imagePath, "请用中文描述这张图片。", 24, 0, false, image);
      System.out.println("\nIMAGE_STATS " + imageStats);
      check(image.text.length() > 0, "No image answer");
      module.resetContext();
      String cancelled = module.generate("从一数到一百。", 32, 0, false, piece -> module.stop());
      check(cancelled.contains("\"stopped\":true"), "stop from callback failed");
      System.out.println("STOP_OK " + cancelled);
      module.resetContext();
      try {
        module.generate("你好", 8, 0, false,
            piece -> { throw new IllegalArgumentException("callback probe"); });
        throw new AssertionError("Callback exception swallowed");
      } catch (IllegalArgumentException expected) {
        check("callback probe".equals(expected.getMessage()), "Wrong callback exception");
      }
      module.resetContext();
      AtomicReference<Throwable> concurrentError = new AtomicReference<>();
      module.generate("你好", 2, 0, false, piece -> {
        Thread t = new Thread(() -> {
          try {
            module.resetContext();
          } catch (Throwable e) {
            concurrentError.set(e);
          }
        });
        t.start();
        try {
          t.join();
        } catch (InterruptedException e) {
          throw new RuntimeException(e);
        }
      });
      check(concurrentError.get() instanceof IllegalStateException, "Concurrent reset accepted");
      System.out.println("CONCURRENCY_OK");
    }
    try (Qwen35MoeModule again = new Qwen35MoeModule(config)) {
      Output answer = new Output();
      again.generateWithImage(imagePath, "请用中文描述图片。", 8, 0, false, piece -> {
        answer.onResult(piece);
        again.close();
      });
      check(answer.text.length() > 0, "Close from callback produced no output");
      System.out.println("CLOSE_CALLBACK_OK");
    }
    System.out.println("\nMOE_JNI_SMOKE_PASS");
  }
}
