package com.example.annoymous;
import android.content.Context;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.Assume;
import org.junit.runner.RunWith;
import java.io.File;
import static org.junit.Assert.assertEquals;
@RunWith(AndroidJUnit4.class)
public class ReconstructionVocabDeviceTest {
 private static native int runWorker(String[] args);
 @Test public void collect() throws Exception {
  String job=InstrumentationRegistry.getArguments().getString("vocab_job");Assume.assumeNotNull(job);
  if(!job.matches("[A-Za-z0-9_-]+"))throw new IllegalArgumentException("Invalid job");
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  String model=InstrumentationRegistry.getArguments().getString("vocab_model");
  if(model==null||!model.matches("[A-Za-z0-9_.-]+"))throw new IllegalArgumentException("Invalid model");
  NativeModel.runtimeInfo();System.loadLibrary("kvvocab");
  assertEquals(0,runWorker(new String[]{ModelFiles.resolve(c,model).getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir,new File(c.getFilesDir(),job).getAbsolutePath()}));
 }
}
