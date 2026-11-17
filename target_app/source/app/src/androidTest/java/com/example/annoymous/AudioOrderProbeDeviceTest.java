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

                                                                                
@RunWith(AndroidJUnit4.class)
public class AudioOrderProbeDeviceTest {
 static native byte[] sendOrdered(long handle,byte[] text,String image,String audio,boolean textFirst);
 @Test public void compare()throws Exception {
  String id=InstrumentationRegistry.getArguments().getString("audio_order_job");Assume.assumeNotNull(id);
  if(!id.matches("audio_order_[A-Za-z0-9_]+"))throw new IOException("Invalid job");
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  File dir=new File(c.getFilesDir(),"kv_jobs/"+id);
  JSONObject job=new JSONObject(new String(Files.readAllBytes(new File(dir,"input.json").toPath()),StandardCharsets.UTF_8));
  long h=0;JSONArray results=new JSONArray();
  try {
   File model=ModelFiles.resolve(c,job.getString("model_file"));
   h=NativeModel.openWithTokenIdScrubbing(model.getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir,false);
   System.loadLibrary("audioorderprobe");
   OmniDialogChatDeviceTest.json(new File(dir,"provenance.json"),new JSONObject()
    .put("model_sha256",OmniDialogChatDeviceTest.sha(model))
    .put("runtime_sha256",OmniDialogChatDeviceTest.sha(new File(c.getApplicationInfo().nativeLibraryDir,"liblitert-lm.so")))
    .put("input_sha256",OmniDialogChatDeviceTest.sha(new File(dir,"input.json")))
    .put("temperature",0).put("top_k",1).put("max_output_tokens",160));
   JSONArray turns=job.getJSONArray("turns");
   String[] names={"multi_audio_first","multi_text_first","single_audio_first","single_text_first"};
   for(int variant=0;variant<names.length;variant++){
    NativeModel.resetWithGenerationSettings(h,0,1,1,160);
    JSONArray records=new JSONArray();boolean multi=variant<2,textFirst=variant%2==1;
    for(int t=multi?0:2;t<3;t++){
     JSONObject turn=turns.getJSONObject(t);String kind=turn.getString("modality");
     File media=kind.equals("text")?null:new File(dir,"media/"+turn.getString("filename"));
     if(media!=null&&(!media.getCanonicalPath().startsWith(new File(dir,"media").getCanonicalPath()+File.separator)
        ||!OmniDialogChatDeviceTest.sha(media).equals(turn.getString("sha256"))))throw new IOException("Bad media");
     JSONObject before=OmniDialogChatDeviceTest.state(h);
     OmniDialogChatDeviceTest.json(new File(dir,"status.json"),new JSONObject().put("state","running").put("variant",names[variant]).put("turn",t+1));
     long tick=SystemClock.elapsedRealtimeNanos();
     byte[] question=turn.getString("text").getBytes(StandardCharsets.UTF_8);
                                                                               
                                                                                
     byte[] reply=kind.equals("audio")&&textFirst
       ?sendOrdered(h,question,"",media.getAbsolutePath(),true)
       :NativeModel.sendMedia(h,question,kind.equals("image")?media.getAbsolutePath():"",kind.equals("audio")?media.getAbsolutePath():"");
     records.put(new JSONObject().put("source_turn",t+1).put("modality",kind).put("question",turn.getString("text"))
       .put("reply",new String(reply,StandardCharsets.UTF_8)).put("before_step",before.getInt("step"))
       .put("seconds",(SystemClock.elapsedRealtimeNanos()-tick)/1e9).put("state",OmniDialogChatDeviceTest.state(h)));
     OmniDialogChatDeviceTest.json(new File(dir,names[variant]+".json"),new JSONObject().put("variant",names[variant]).put("turns",records));
    }
    results.put(new JSONObject().put("variant",names[variant]).put("turns",records));
   }
   OmniDialogChatDeviceTest.json(new File(dir,"result.json"),new JSONObject().put("variants",results));
   OmniDialogChatDeviceTest.json(new File(dir,"status.json"),new JSONObject().put("state","complete"));
  }catch(Exception e){OmniDialogChatDeviceTest.json(new File(dir,"status.json"),new JSONObject().put("state","failed").put("error",e.toString()));throw e;}
  finally{if(h!=0)NativeModel.close(h);}
 }
}
