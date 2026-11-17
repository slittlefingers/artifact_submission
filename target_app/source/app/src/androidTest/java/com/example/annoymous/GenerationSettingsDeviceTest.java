package com.example.annoymous;

import android.content.Context;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.nio.charset.StandardCharsets;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class GenerationSettingsDeviceTest {
 @Test public void preferencesRoundTripAndValidation(){
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  GenerationSettings old=GenerationSettings.load(c);
  try{
   new GenerationSettings(.7f,40,.9f,256).save(c);
   GenerationSettings actual=GenerationSettings.load(c);
   assertEquals(.7f,actual.temperature,0);assertEquals(40,actual.topK);assertEquals(.9f,actual.topP,0);assertEquals(256,actual.maxOutputTokens);
   try{new GenerationSettings(Float.NaN,1,1,512);fail();}catch(IllegalArgumentException expected){}
   try{new GenerationSettings(1,0,1,512);fail();}catch(IllegalArgumentException expected){}
   try{new GenerationSettings(1,1,0,512);fail();}catch(IllegalArgumentException expected){}
   try{new GenerationSettings(1,1,1,0);fail();}catch(IllegalArgumentException expected){}
  }finally{old.save(c);}
 }
 @Test public void ministralUsesConfiguredLimitsAndSampler()throws Exception{
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  long h=NativeModel.openWithBackend(ModelFiles.resolve(c,"ministral3_3b_vl_Google_Tensor_G5.litertlm").getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir,"npu",true);
  try{
   byte[] prompt="Write a detailed explanation of how rain forms, in at least ten complete sentences.".getBytes(StandardCharsets.UTF_8);
   new GenerationSettings(0,40,1,1).startSession(h);
   String shortReply=new String(NativeModel.sendMedia(h,prompt,"",""),StandardCharsets.UTF_8);
   new GenerationSettings(0,40,1,128).startSession(h);
   String longReply=new String(NativeModel.sendMedia(h,prompt,"",""),StandardCharsets.UTF_8);
   assertTrue("Short="+shortReply+" long="+longReply,longReply.length()>shortReply.length()+200);
   new GenerationSettings(.7f,40,.9f,16).startSession(h);
   assertTrue(NativeModel.sendMedia(h,"Say hello.".getBytes(StandardCharsets.UTF_8),"","").length>0);
   try{NativeModel.resetWithGenerationSettings(h,1,1,1,0);fail();}catch(IllegalArgumentException expected){}
  }finally{NativeModel.close(h);}
 }
}
