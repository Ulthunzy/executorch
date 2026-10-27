import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONObject;
import org.pytorch.executorch.extension.llm.Qwen35DenseModule;

/** Real hybrid PTE test; run through app_process with model directory, libs, and JPG. */
public final class Qwen35DenseSmokeTest {
  private static void check(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
  private static String run(Qwen35DenseModule model, String image, String prompt, int count,
      boolean thinking, StringBuilder output) throws Exception {
    String stats = model.generateWithImage(image, prompt, count, 0, thinking, output::append);
    JSONObject json = new JSONObject(stats);
    check(json.getInt("generated_tokens") <= count, "Output budget exceeded");
    check(json.getDouble("model_load_end_ms") > json.getDouble("model_load_start_ms"), "Missing load stats");
    System.out.println("DENSE_RESULT " + output + "\nDENSE_STATS " + stats);
    return stats;
  }
  public static void main(String[] args) throws Exception {
    Thread.setDefaultUncaughtExceptionHandler((thread, error) -> {
      error.printStackTrace(); System.exit(1);
    });
    System.out.println("DENSE_STAGE start");
    Class<?> stream = Class.forName("org.pytorch.executorch.extension.llm.Qwen35DenseModule$Stream");
    Constructor<?> constructor = stream.getDeclaredConstructor(Qwen35DenseModule.Callback.class);
    constructor.setAccessible(true);
    StringBuilder unicode = new StringBuilder();
    Object decoder = constructor.newInstance((Qwen35DenseModule.Callback) unicode::append);
    Method bytes = stream.getDeclaredMethod("onBytes", byte[].class);
    Method finish = stream.getDeclaredMethod("finish");
    bytes.setAccessible(true); finish.setAccessible(true);
    for (byte b : "中文😀".getBytes(StandardCharsets.UTF_8)) bytes.invoke(decoder, (Object) new byte[]{b});
    finish.invoke(decoder);
    check(unicode.toString().equals("中文😀"), "UTF-8 stream corruption");

    System.out.println("DENSE_STAGE unicode_pass");
    Qwen35DenseModule.Config config = new Qwen35DenseModule.Config(args[0], args[1]);
    final Qwen35DenseModule model = new Qwen35DenseModule(config);
    System.out.println("DENSE_STAGE constructed");
    try {
      StringBuilder first = new StringBuilder();
      JSONObject a = new JSONObject(run(model, "", "1+1等于多少？只回答数字。", 16, false, first));
      check(a.getInt("generated_tokens") < 16, "EOS did not stop generation");
      JSONObject b = new JSONObject(run(model, "", "2+2等于多少？只回答数字。", 8, false, new StringBuilder()));
      check(b.getInt("global_kv_tokens") > a.getInt("global_kv_tokens"), "History was discarded");
      model.resetContext();
      StringBuilder repeated = new StringBuilder();
      JSONObject reset = new JSONObject(run(model, "", "1+1等于多少？只回答数字。", 16, false, repeated));
      check(first.toString().equals(repeated.toString()), "Reset did not restore deterministic state");
      check(reset.getLong("model_load_end_ms") == a.getLong("model_load_end_ms"), "Reset reloaded the model");
      model.resetContext();
      JSONObject one = new JSONObject(run(model, "", "介绍一下自己。", 1, false, new StringBuilder()));
      check(one.getInt("generated_tokens") == 1, "First token was not counted");
      model.resetContext();
      StringBuilder image = new StringBuilder();
      run(model, args[2], "请用中文描述这张图片中的人物和动物。", 48, false, image);
      check(image.length() > 0, "No image response");
      model.resetContext();
      StringBuilder thoughts = new StringBuilder();
      run(model, "", "9乘以8等于多少？", 16, true, thoughts);
      check(thoughts.toString().startsWith("<think>"), "Thinking marker missing");
      model.resetContext();
      JSONObject stopped = new JSONObject(model.generate("介绍一下相对论。", 128, 0, false, piece -> model.stop()));
      check(stopped.getInt("generated_tokens") == 1, "stop() did not stop at token boundary");
      // A new request after a decode stop continues without unloading the session.
      run(model, "", "1+1等于多少？", 8, false, new StringBuilder());
      model.resetContext();
      StringBuilder sampled = new StringBuilder();
      model.generate("你好。", 3, 0.7f, false, sampled::append);
      check(sampled.length() > 0, "Temperature sampling produced no output");
      try {
        model.generate("你好。", 0, 0, false, x -> {});
        throw new AssertionError("Zero token budget accepted");
      } catch (IllegalArgumentException expected) {}
      StringBuilder oversized = new StringBuilder();
      for (int i = 0; i < 2048; ++i) oversized.append("你好世界。");
      try {
        model.generate(oversized.toString(), 1, 0, false, x -> {});
        throw new AssertionError("Oversized prompt accepted");
      } catch (RuntimeException expected) {}
      run(model, "", "1+1等于多少？", 8, false, new StringBuilder());
      model.resetContext();
      try {
        model.generate("你好。", 4, 0, false, x -> { throw new IllegalStateException("callback probe"); });
        throw new AssertionError("Callback exception was swallowed");
      } catch (IllegalStateException expected) {}
      model.resetContext();
      CountDownLatch callback = new CountDownLatch(1);
      CountDownLatch closing = new CountDownLatch(1);
      AtomicReference<Throwable> failure = new AtomicReference<>();
      Thread worker = new Thread(() -> {
        try {
          model.generate("请详细介绍相对论。", 128, 0, false, s -> {
            callback.countDown();
            try {
              if (!closing.await(10, TimeUnit.SECONDS))
                throw new AssertionError("close did not start");
            } catch (InterruptedException e) { throw new AssertionError(e); }
            model.stop();
          });
        }
        catch (Throwable e) { failure.set(e); }
      });
      worker.start();
      check(callback.await(45, TimeUnit.SECONDS), "No callback from worker");
      try {
        model.generate("并发调用", 1, 0, false, s -> {});
        throw new AssertionError("Concurrent generate was accepted");
      } catch (IllegalStateException expected) {}
      Thread closer = new Thread(model::close);
      closer.start();
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
      try {
        while (closer.getState() != Thread.State.WAITING && System.nanoTime() < deadline)
          Thread.sleep(1);
        check(closer.getState() == Thread.State.WAITING, "close did not wait for generation");
      } finally { closing.countDown(); }
      closer.join(10000);
      check(!closer.isAlive(), "Concurrent close did not finish");
      worker.join(10000);
      check(!worker.isAlive() && failure.get() == null, "close()/stop() failed: " + failure.get());
      model.close();
      try { model.resetContext(); throw new AssertionError("Closed session was accepted"); }
      catch (IllegalStateException expected) {}
    } finally { model.close(); }
    System.out.println("DENSE_JNI_SMOKE_PASS");
    System.exit(0);
  }
}
