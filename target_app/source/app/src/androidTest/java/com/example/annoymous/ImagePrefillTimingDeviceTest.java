package com.example.annoymous;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.json.*;
import java.io.File;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ImagePrefillTimingDeviceTest {
 @Test public void qwenImagePrefill() throws Exception {
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  String job=InstrumentationRegistry.getArguments().getString("timing_job");
  assertNotNull(job);assertTrue(job.matches("[A-Za-z0-9_-]+"));
  JSONObject input=new JSONObject(new String(Files.readAllBytes(new File(c.getFilesDir(),"kv_jobs/"+job+"/input.json").toPath()),StandardCharsets.UTF_8));
  JSONArray samples=input.getJSONArray("samples"),rows=new JSONArray();
  String model=InstrumentationRegistry.getArguments().getString("timing_model","qwen3_5_2b_vl_Google_Tensor_G5.litertlm");
  long tick=SystemClock.elapsedRealtimeNanos();
  long h=NativeModel.open(ModelFiles.resolve(c,model).getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir);
  double load=(SystemClock.elapsedRealtimeNanos()-tick)/1e9;
  try {
   for(int i=0;i<Math.min(2,samples.length());i++) {
    if(i>0)NativeModel.reset(h);
    JSONObject sample=samples.getJSONObject(i);
    File image=new File(c.getFilesDir(),"vqav2_eval_images/"+sample.getString("image_name"));assertTrue(image.isFile());
    tick=SystemClock.elapsedRealtimeNanos();
    byte[] raw=NativeModel.prefillImageForCapture(h,sample.getString("user_prompt").getBytes(StandardCharsets.UTF_8),image.getAbsolutePath());
    double seconds=(SystemClock.elapsedRealtimeNanos()-tick)/1e9;
    JSONObject metadata=new JSONObject(new String(raw,StandardCharsets.UTF_8));
    assertTrue(metadata.getInt("kv_positions")>0);
    JSONObject row=new JSONObject().put("id",sample.getString("id")).put("image_name",image.getName()).put("prompt",sample.getString("user_prompt")).put("prefill_seconds",seconds).put("metadata",metadata);
    rows.put(row);Log.i("ImagePrefillTiming","sample="+sample.getString("id")+" seconds="+seconds+" positions="+metadata.getInt("kv_positions"));
    Files.write(new File(c.getFilesDir(),"qwen_image_prefill_benchmark.json").toPath(),new JSONObject().put("model",model).put("load_seconds",load).put("generated_tokens",0).put("samples",rows).toString(2).getBytes(StandardCharsets.UTF_8));
   }
  } finally {NativeModel.close(h);}
 }
}
