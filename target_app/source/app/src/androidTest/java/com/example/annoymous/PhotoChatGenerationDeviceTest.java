package com.example.annoymous;

import android.content.Context;
import android.os.SystemClock;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.Assume;
import org.junit.runner.RunWith;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;

                                                                                          
@RunWith(AndroidJUnit4.class)
public class PhotoChatGenerationDeviceTest {
 private static JSONObject state(long h)throws Exception {
  JSONObject value=new JSONObject(new String(NativeModel.inspect(h,"{\"layer\":4}"),StandardCharsets.UTF_8));
  JSONArray layers=value.optJSONArray("layers");
  if(layers!=null)for(int i=0;i<layers.length();i++){
   JSONArray buffers=layers.getJSONObject(i).getJSONArray("buffers");
   for(int j=0;j<buffers.length();j++)buffers.getJSONObject(j).remove("samples");
  }
  return value;
 }
 private static void json(File path,JSONObject value)throws Exception {
  File tmp=new File(path+".tmp");Files.write(tmp.toPath(),value.toString(2).getBytes(StandardCharsets.UTF_8));
  Files.move(tmp.toPath(),path.toPath(),StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
 }
 private static String sha(File file)throws Exception {
  MessageDigest d=MessageDigest.getInstance("SHA-256");byte[] b=new byte[1048576];
  try(InputStream in=new FileInputStream(file)){int n;while((n=in.read(b))!=-1)d.update(b,0,n);}
  StringBuilder s=new StringBuilder();for(byte x:d.digest())s.append(String.format(java.util.Locale.ROOT,"%02x",x&255));return s.toString();
 }
 private static void status(File dir,String state,int done,int total,String id,int turn)throws Exception {
  json(new File(dir,"status.json"),new JSONObject().put("state",state).put("completed",done).put("total",total).put("sample_id",id).put("turn",turn).put("updated_ms",System.currentTimeMillis()));
 }
 @Test public void generateBatch()throws Exception {
  String jobId=InstrumentationRegistry.getArguments().getString("photochat_job");Assume.assumeNotNull(jobId);
  if(!jobId.matches("[A-Za-z0-9_-]{1,64}"))throw new IllegalArgumentException("Invalid job");
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  File dir=new File(c.getFilesDir(),"kv_jobs/"+jobId);
  JSONObject job=new JSONObject(new String(Files.readAllBytes(new File(dir,"input.json").toPath()),StandardCharsets.UTF_8));
  JSONArray samples=job.getJSONArray("samples");long h=0;int done=0;String active="";
  try {
   status(dir,"loading",0,samples.length(),"",0);
   File model=ModelFiles.resolve(c,job.getString("model_file"));
   JSONObject provenance=new JSONObject().put("model_file",model.getName()).put("model_sha256",sha(model))
    .put("runtime_sha256",sha(new File(c.getApplicationInfo().nativeLibraryDir,"liblitert-lm.so")))
    .put("bridge_sha256",sha(new File(c.getApplicationInfo().nativeLibraryDir,"libannoymous.so")))
    .put("input_sha256",sha(new File(dir,"input.json"))).put("backend","npu").put("device",android.os.Build.MODEL)
    .put("settings",job.getJSONObject("settings")).put("assistant_source","live_target_model_generation")
    .put("same_session_between_turns",true).put("new_session_between_samples",true)
    .put("content_order","image before text within the image-bearing user turn")
    .put("capture_stage","after_second_response").put("layer",4)
    .put("window_note","The pending last generated token may not yet have a KV entry.");
   h=NativeModel.open(model.getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir);
   json(new File(dir,"job.json"),provenance);
   JSONObject settings=job.getJSONObject("settings");
   for(int i=0;i<samples.length();i++){
    if(new File(dir,"cancel").exists())throw new IOException("Cancelled by host");
    JSONObject sample=samples.getJSONObject(i);active=sample.getString("id");
    if(!active.matches("photochat_[0-9]{4}"))throw new IOException("Invalid sample ID");
    File out=new File(dir,active);out.mkdirs();
    if(new File(out,"complete.json").isFile()){done++;continue;}
    File image=new File(dir,"images/"+sample.getString("image_name"));
    if(!image.getCanonicalPath().startsWith(new File(dir,"images").getCanonicalPath()+File.separator)||!sha(image).equals(sample.getString("image_sha256")))throw new IOException("Image checksum mismatch");
    NativeModel.resetWithGenerationSettings(h,(float)settings.getDouble("temperature"),settings.getInt("top_k"),(float)settings.getDouble("top_p"),settings.getInt("max_output_tokens"));
    JSONObject reset=state(h);
    if(reset.getInt("kv_positions")!=0)throw new IOException("Session position did not reset");
    JSONArray turns=sample.getJSONArray("user_turns"),messages=new JSONArray();
    if(turns.length()!=2)throw new IOException("Expected two user turns");
    for(int t=0;t<2;t++){
     if(new File(dir,"cancel").exists())throw new IOException("Cancelled by host");
     JSONObject turn=turns.getJSONObject(t);String text=turn.getString("text");
     if(text.isEmpty()||turn.has("assistant"))throw new IOException("Invalid user-only input");
     status(dir,"generating",done,samples.length(),active,t+1);
     long tick=SystemClock.elapsedRealtimeNanos();
     String reply=new String(NativeModel.sendMedia(h,text.getBytes(StandardCharsets.UTF_8),turn.getBoolean("has_image")?image.getAbsolutePath():"",""),StandardCharsets.UTF_8);
     if(reply.trim().isEmpty())throw new IOException("Empty model reply");
     JSONObject state=state(h);
     JSONObject record=new JSONObject().put("turn",t+1).put("user",turn).put("assistant",reply)
      .put("assistant_source","live_target_model_generation").put("generation_seconds",(SystemClock.elapsedRealtimeNanos()-tick)/1e9).put("state",state);
     json(new File(out,"turn"+(t+1)+".json"),record);messages.put(record);
    }
    status(dir,"capturing",done,samples.length(),active,2);
    JSONObject capture=BatchKvCaptureDeviceTest.capture(h,new File(out,"layer4.gkv.gz"),4);
    JSONObject result=new JSONObject().put("id",active).put("group",sample.getString("group")).put("image_sha256",sha(image))
     .put("turns",messages).put("settings",settings).put("kv",capture).put("layer",4).put("capture_stage","after_second_response")
     .put("assistant_source","live_target_model_generation").put("reset_state",reset);
    json(new File(out,"sample.json"),result);json(new File(out,"complete.json"),new JSONObject().put("id",active).put("sample_sha256",sha(new File(out,"sample.json"))));
    done++;status(dir,"running",done,samples.length(),active,2);
   }
   status(dir,"done",done,samples.length(),active,2);
  }catch(Exception e){json(new File(dir,"status.json"),new JSONObject().put("state","failed").put("completed",done).put("sample_id",active).put("error",e.toString()));throw e;}
  finally{if(h!=0)NativeModel.close(h);}
 }
}
