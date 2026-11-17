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
public class OmniDialogChatDeviceTest {
 static void json(File p,JSONObject v)throws Exception {
  File tmp=new File(p+".tmp");Files.write(tmp.toPath(),v.toString(2).getBytes(StandardCharsets.UTF_8));
  Files.move(tmp.toPath(),p.toPath(),StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
 }
 static String sha(File f)throws Exception {
  MessageDigest d=MessageDigest.getInstance("SHA-256");byte[] b=new byte[1048576];
  try(InputStream s=new FileInputStream(f)){int n;while((n=s.read(b))!=-1)d.update(b,0,n);}
  StringBuilder r=new StringBuilder();for(byte x:d.digest())r.append(String.format(java.util.Locale.ROOT,"%02x",x&255));return r.toString();
 }
 static JSONObject state(long h)throws Exception {
  JSONObject v=new JSONObject(new String(NativeModel.inspect(h,"{\"layer\":4}"),StandardCharsets.UTF_8));
  JSONObject labels=new JSONObject(new String(NativeModel.inspect(h,"{\"layer\":-1,\"slot\":-1,\"format\":\"inspect\"}"),StandardCharsets.UTF_8));
  v.put("tokens",labels.getJSONArray("tokens"));
  JSONArray a=v.getJSONArray("layers");for(int i=0;i<a.length();i++){
   JSONArray b=a.getJSONObject(i).getJSONArray("buffers");for(int j=0;j<b.length();j++)b.getJSONObject(j).remove("samples");
  }return v;
 }
 @Test public void collect()throws Exception {
  String id=InstrumentationRegistry.getArguments().getString("omni_job");Assume.assumeNotNull(id);
  if(!id.matches("[A-Za-z0-9_-]{1,64}"))throw new IOException("Invalid job ID");
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();File dir=new File(c.getFilesDir(),"kv_jobs/"+id);
  JSONObject job=new JSONObject(new String(Files.readAllBytes(new File(dir,"input.json").toPath()),StandardCharsets.UTF_8));
  JSONArray rows=job.getJSONArray("samples");long h=0;int done=0;String sid="";
  try {
   json(new File(dir,"status.json"),new JSONObject().put("state","loading"));
   File model=ModelFiles.resolve(c,job.getString("model_file"));
                                                                               
   h=NativeModel.openWithTokenIdScrubbing(model.getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir,false);
   json(new File(dir,"job.json"),new JSONObject().put("model_sha256",sha(model)).put("input_sha256",sha(new File(dir,"input.json")))
    .put("token_id_scrubbing",false).put("token_ids_source","Runtime token buffer, not retokenized text")
    .put("backend","npu").put("capture_stage","after_each_reply").put("same_session_between_turns",true)
    .put("runtime_sha256",sha(new File(c.getApplicationInfo().nativeLibraryDir,"liblitert-lm.so"))));
   for(int i=0;i<rows.length();i++){
    JSONObject sample=rows.getJSONObject(i);sid=sample.getString("id");
    if(!sid.matches("omni_[0-9]{4}"))throw new IOException("Invalid sample ID");
    File out=new File(dir,sid);out.mkdirs();JSONArray turns=sample.getJSONArray("turns"),records=new JSONArray();
    if(turns.length()<2||turns.length()>6)throw new IOException("Expected 2-6 turns");
    NativeModel.resetWithGenerationSettings(h,0,1,1,job.getInt("max_output_tokens"));
    if(state(h).getInt("kv_positions")!=0)throw new IOException("Session did not reset");
    for(int t=0;t<turns.length();t++){
     if(new File(dir,"cancel").exists())throw new IOException("Cancelled");
     JSONObject turn=turns.getJSONObject(t);String kind=turn.getString("modality");
     if(!kind.equals("audio")&&!kind.equals("image")&&!kind.equals("text"))throw new IOException("Invalid modality");
     File media=kind.equals("text")?null:new File(dir,"media/"+turn.getString("filename"));
     if(media!=null&&(!media.getCanonicalPath().startsWith(new File(dir,"media").getCanonicalPath()+File.separator)||!sha(media).equals(turn.getString("sha256"))))throw new IOException("Bad media");
     JSONObject before=state(h);
     json(new File(dir,"status.json"),new JSONObject().put("state","generating").put("completed",done).put("sample",sid).put("turn",t+1));
     long tick=SystemClock.elapsedRealtimeNanos();
     String reply=new String(NativeModel.sendMedia(h,turn.getString("text").getBytes(StandardCharsets.UTF_8),kind.equals("image")?media.getAbsolutePath():"",kind.equals("audio")?media.getAbsolutePath():""),StandardCharsets.UTF_8);
     double seconds=(SystemClock.elapsedRealtimeNanos()-tick)/1e9;JSONObject after=state(h);
     if(after.getInt("step")<=before.getInt("step")||after.getInt("step")>=4000)throw new IOException("Invalid context size");
     JSONObject captures=new JSONObject();
     for(int layer:new int[]{0,4,9,14})captures.put(""+layer,BatchKvCaptureDeviceTest.capture(h,new File(out,"turn"+(t+1)+"_layer"+layer+".gkv.gz"),layer));
     JSONObject record=new JSONObject().put("turn",t+1).put("user",turn).put("assistant",reply).put("assistant_source","live_target_model_generation")
      .put("before_step",before.getInt("step")).put("state",after).put("generation_seconds",seconds).put("kv",captures);
     json(new File(out,"turn"+(t+1)+".json"),record);records.put(record);
    }
    JSONObject result=new JSONObject().put("id",sid).put("turns",records).put("final_step",records.getJSONObject(records.length()-1).getJSONObject("state").getInt("step"));
    json(new File(out,"sample.json"),result);json(new File(out,"complete.json"),new JSONObject().put("sample_sha256",sha(new File(out,"sample.json"))));
    done++;
   }
   json(new File(dir,"status.json"),new JSONObject().put("state","complete").put("completed",done));
  }catch(Exception e){json(new File(dir,"status.json"),new JSONObject().put("state","failed").put("completed",done).put("sample",sid).put("error",e.toString()));throw e;}
  finally{if(h!=0)NativeModel.close(h);}
 }
}
