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
public class AllLayerCaptureDeviceTest {
 private static void json(File path,JSONObject value)throws Exception {
  File tmp=new File(path+".tmp");Files.write(tmp.toPath(),value.toString(2).getBytes(StandardCharsets.UTF_8));
  Files.move(tmp.toPath(),path.toPath(),StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
 }
 private static void status(File dir,String state,int done,int total,String id)throws Exception {
  json(new File(dir,"status.json"),new JSONObject().put("state",state).put("completed",done).put("total",total).put("sample_id",id).put("updated_ms",System.currentTimeMillis()));
 }
 private static String sha(File file)throws Exception {
  MessageDigest h=MessageDigest.getInstance("SHA-256");byte[] b=new byte[1024*1024];
  try(InputStream in=new FileInputStream(file)){int n;while((n=in.read(b))!=-1)h.update(b,0,n);}
  StringBuilder s=new StringBuilder();for(byte x:h.digest())s.append(String.format(java.util.Locale.ROOT,"%02x",x&255));return s.toString();
 }
 private static void cancel(File dir)throws IOException {if(new File(dir,"cancel").exists())throw new IOException("Cancelled by host");}
 @Test public void captureBatch()throws Exception {
  String id=InstrumentationRegistry.getArguments().getString("all_layer_job");Assume.assumeNotNull(id);
  if(!id.matches("[A-Za-z0-9_-]{1,64}"))throw new IllegalArgumentException("Invalid job ID");
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();File dir=new File(c.getFilesDir(),"kv_jobs/"+id);
  JSONObject job=new JSONObject(new String(Files.readAllBytes(new File(dir,"input.json").toPath()),StandardCharsets.UTF_8));JSONArray samples=job.getJSONArray("samples");
  int selectedLayer=job.optInt("selected_layer",-1);
  long h=0;int done=0;String current="";
  try {
   status(dir,"loading",0,samples.length(),"");
   String modelName=job.optString("model_file","gemma-4-E2B-it_Google_Tensor_G5.litertlm");
   if(!modelName.matches("[A-Za-z0-9_.-]+\\.litertlm"))throw new IllegalArgumentException("Invalid model file");
   File model=ModelFiles.resolve(c,modelName);
   if(!model.isFile())throw new FileNotFoundException("Requested model missing");
   JSONObject info=new JSONObject().put("schema_version",1).put("model",job.optString("model","gemma4-e2b")).put("model_file",model.getName())
    .put("model_sha256",sha(model))
    .put("vendor_sha256",sha(new File(c.getApplicationInfo().nativeLibraryDir,"libedgetpu_litert.so"))).put("runtime_sha256",sha(new File(c.getApplicationInfo().nativeLibraryDir,"liblitert-lm.so")))
    .put("bridge_sha256",sha(new File(c.getApplicationInfo().nativeLibraryDir,"libannoymous.so")))
    .put("capture_stage","after_prefill").put("layer",-1).put("generated_tokens",0).put("session_per_sample",true)
    .put("token_id_scrubbing",true).put("device",android.os.Build.MODEL).put("android_fingerprint",android.os.Build.FINGERPRINT)
    .put("input_sha256",sha(new File(dir,"input.json"))).put("backend","npu");
   h=NativeModel.open(model.getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir);
   json(new File(dir,"job.json"),info);
   for(int i=0;i<samples.length();i++){
    cancel(dir);JSONObject sample=samples.getJSONObject(i);current=sample.getString("id");
    String prompt=sample.getString("user_prompt");
    if(!current.matches("[A-Za-z0-9_-]{1,64}")||prompt.isEmpty()||prompt.codePointCount(0,prompt.length())>=100||prompt.indexOf('\0')>=0)throw new IllegalArgumentException("Invalid sample");
    File out=new File(dir,current);if(!out.mkdir())throw new IOException("Existing sample output");
    if(i>0)NativeModel.reset(h);
    String formatted=job.optString("prefix","<|turn>user\n")+prompt+job.optString("suffix","<turn|>\n<|turn>model\n");
    status(dir,"prefilling",done,samples.length(),current);long tick=SystemClock.elapsedRealtimeNanos();
    int[] tokens=NativeModel.prefillForCapture(h,formatted.getBytes(StandardCharsets.UTF_8));
    if(sample.has("expected_input_token_ids")){
    JSONArray expectedIds=sample.getJSONArray("expected_input_token_ids");
    if(tokens.length!=expectedIds.length())throw new IOException("Token count changed");
    for(int q=0;q<tokens.length;q++)if(tokens[q]!=expectedIds.getInt(q))throw new IOException("Token ID changed at "+q);
    }
    double seconds=(SystemClock.elapsedRealtimeNanos()-tick)/1e9;
    JSONObject state=new JSONObject(new String(NativeModel.inspect(h,new JSONObject().put("layer",selectedLayer).toString()),StandardCharsets.UTF_8));
    json(new File(dir,"layer_structure.json"),new JSONObject(new String(NativeModel.inspect(h,"{\"layer\":-1}"),StandardCharsets.UTF_8)));
    if(state.getInt("kv_positions")!=tokens.length||tokens.length>512)throw new IOException("Input/KV position count mismatch: "+tokens.length+" vs "+state.optInt("kv_positions"));
    JSONArray layers=state.getJSONArray("layers");
    for(int z=0;z<layers.length();z++){
      JSONArray buffers=layers.getJSONObject(z).getJSONArray("buffers");
      for(int n=0;n<buffers.length();n++)if(buffers.getJSONObject(n).getInt("capacity")<tokens.length)throw new IOException("Prompt would be evicted");
    }
    cancel(dir);status(dir,"capturing",done,samples.length(),current);
    File part=new File(out,"all_layers.gkv.gz.part"),dest=new File(out,"all_layers.gkv.gz");
    JSONObject capture=BatchKvCaptureDeviceTest.capture(h,part,selectedLayer);Files.move(part.toPath(),dest.toPath(),StandardCopyOption.ATOMIC_MOVE);
    JSONArray ids=new JSONArray();for(int token:tokens)ids.put(token);
    JSONObject meta=new JSONObject().put("id",current).put("conversation_id",sample.getString("conversation_id"))
     .put("user_prompt",prompt).put("formatted_input",formatted).put("input_token_ids",ids).put("input_tokens",tokens.length)
     .put("capture_stage","after_prefill").put("layer",-1).put("generated_tokens",0).put("prefill_seconds",seconds).put("kv",capture)
     .put("model_file",modelName).put("bos_source","Runtime start-token metadata; included in input_token_ids").put("frontier_flush","One newline prefill processes the original final token; its added newline remains pending without KV.");
    json(new File(out,"sample.json"),meta);
    json(new File(dir,"ready.json"),new JSONObject().put("id",current).put("index",i));status(dir,"awaiting_host",done,samples.length(),current);
    File ack=new File(dir,"ack-"+current);long deadline=SystemClock.elapsedRealtime()+600000;
    while(!ack.isFile()){cancel(dir);if(SystemClock.elapsedRealtime()>deadline)throw new IOException("Host acknowledgement timed out");Thread.sleep(100);}
    for(File f:out.listFiles())if(!f.delete())throw new IOException("Cannot clean acknowledged sample");
    if(!out.delete()||!ack.delete()||!new File(dir,"ready.json").delete())throw new IOException("Cannot finish acknowledgement");
    done++;status(dir,"running",done,samples.length(),current);
   }
   status(dir,"done",done,samples.length(),current);
  }catch(Exception e){json(new File(dir,"status.json"),new JSONObject().put("state","failed").put("completed",done).put("error",e.toString()));throw e;}
  finally{if(h!=0)NativeModel.close(h);}
 }
}
