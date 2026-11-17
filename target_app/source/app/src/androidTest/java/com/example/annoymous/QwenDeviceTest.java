package com.example.annoymous;
import android.content.Context;
import android.os.ParcelFileDescriptor;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import static org.junit.Assert.*;
@RunWith(AndroidJUnit4.class)
public class QwenDeviceTest {
 private JSONObject read(long h,int layer,int slot)throws Exception{return new JSONObject(new String(NativeModel.inspect(h,new JSONObject().put("layer",layer).put("slot",slot).toString()),StandardCharsets.UTF_8));}
 private String send(long h,String text){return new String(NativeModel.sendMedia(h,text.getBytes(StandardCharsets.UTF_8),"",""),StandardCharsets.UTF_8).trim();}
 @Test public void longRepliesWithDescriptorReuse() throws Exception {
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  File model=ModelFiles.resolve(c,"qwen3_5_2b_vl_Google_Tensor_G5.litertlm");
  java.util.concurrent.atomic.AtomicBoolean running=new java.util.concurrent.atomic.AtomicBoolean(true);
  java.util.concurrent.atomic.AtomicReference<Throwable> churnError=new java.util.concurrent.atomic.AtomicReference<>();
  Thread churn=new Thread(()->{try{while(running.get()){
   ParcelFileDescriptor[] descriptors=new ParcelFileDescriptor[32];
   try {for(int i=0;i<descriptors.length;i++)descriptors[i]=ParcelFileDescriptor.open(new File("/dev/null"),ParcelFileDescriptor.MODE_READ_ONLY);Thread.sleep(2);}
   finally {for(ParcelFileDescriptor fd:descriptors)if(fd!=null)fd.close();}
  }}catch(Throwable t){churnError.set(t);}},"fence-fd-reuse-test");
  churn.start();
  try {for(boolean scrub:new boolean[]{false,true}){
   long h=NativeModel.openWithTokenIdScrubbing(model.getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir,scrub);
   try {
    String first=send(h,"Explain how rain forms. Give a detailed answer in several sentences.");
    assertTrue("First reply: "+first,first.length()>80);
    String second=send(h,"Continue with a detailed explanation of how snow forms.");
    assertTrue("Second reply: "+second,second.length()>80);
    assertTrue(read(h,-1,-1).getInt("step")>100);
    NativeModel.reset(h);
    assertEquals("5",send(h,"What is 2 + 3? Reply with the number only."));
   }finally{NativeModel.close(h);}
  }}finally{running.set(false);churn.join(5000);}
  if(churnError.get()!=null)throw new AssertionError("Descriptor churn failed",churnError.get());
 }
 @Test public void qwenModesAndLinearDebug() throws Exception {
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();
  File model=ModelFiles.resolve(c,"qwen3_5_2b_vl_Google_Tensor_G5.litertlm");assertEquals(5543268480L,model.length());
  for(boolean scrub:new boolean[]{false,true}){
   long h=NativeModel.openWithTokenIdScrubbing(model.getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir,scrub);
   try{
    assertTrue(NativeModel.runtimeInfo(),NativeModel.runtimeInfo().contains("vendor_api=0.21.0"));assertTrue(NativeModel.runtimeInfo(),NativeModel.runtimeInfo().contains(c.getApplicationInfo().nativeLibraryDir));
    assertEquals("5",send(h,"What is 2 + 3? Reply with the number only."));
    JSONObject data=read(h,-1,-1);JSONArray layers=data.getJSONArray("layers");assertEquals(24,layers.length());int linear=0,full=0,linearId=-1,fullId=-1;
    for(int i=0;i<layers.length();i++){JSONObject layer=layers.getJSONObject(i);if(layer.getString("attention_type").equals("linear")){linear++;linearId=layer.getInt("layer");}else{full++;fullId=layer.getInt("layer");}}assertEquals(18,linear);assertEquals(6,full);
    JSONArray tokens=data.getJSONArray("tokens"),allocations=data.getJSONArray("token_allocations");int positive=0;
    for(int i=0;i<tokens.length();i++){JSONObject t=tokens.getJSONObject(i);int size=allocations.getJSONObject(t.getInt("candidate")).getInt("size");if(t.getString("region").equals("live")&&t.getInt("index")<size-1&&t.getInt("id")>0)positive++;}if(scrub)assertEquals("Consumed Qwen IDs",0,positive);else assertTrue(positive>0);
    JSONArray buffers=read(h,linearId,0).getJSONArray("layers").getJSONObject(0).getJSONArray("buffers");assertEquals(2,buffers.length());boolean r=false,conv=false;
    for(int i=0;i<buffers.length();i++){JSONObject b=buffers.getJSONObject(i),v=b.getJSONArray("samples").getJSONObject(0);assertEquals(-1,b.getInt("sequence_axis"));assertEquals(-1,v.getInt("position"));assertTrue(v.getBoolean("full"));if(b.getString("state_kind").equals("recurrent")){r=true;assertEquals(128,v.getInt("elements"));assertEquals(2048,b.getInt("capacity"));}else{conv=true;assertEquals(3,v.getInt("elements"));assertEquals(6144,b.getInt("capacity"));}}assertTrue(r&&conv);
    JSONArray kv=read(h,fullId,0).getJSONArray("layers").getJSONObject(0).getJSONArray("buffers");assertEquals(2,kv.length());for(int i=0;i<kv.length();i++){JSONObject b=kv.getJSONObject(i);assertEquals("kv",b.getString("state_kind"));assertEquals(b.getString("name").contains("_v_")?3:2,b.getInt("sequence_axis"));assertEquals(2048,b.getInt("capacity"));}
    File raw=new File(c.getCacheDir(),"qwen-state-test.gkv");try{
     try(ParcelFileDescriptor fd=ParcelFileDescriptor.open(raw,ParcelFileDescriptor.MODE_CREATE|ParcelFileDescriptor.MODE_TRUNCATE|ParcelFileDescriptor.MODE_READ_WRITE)){NativeModel.inspect(h,new JSONObject().put("layer",linearId).put("format","kv_raw").put("fd",fd.getFd()).toString());}
     byte[] bytes=Files.readAllBytes(raw.toPath());assertEquals("GHOSTKV1",new String(bytes,0,8,StandardCharsets.US_ASCII));long length=ByteBuffer.wrap(bytes,8,8).order(ByteOrder.LITTLE_ENDIAN).getLong();JSONArray desc=new JSONObject(new String(bytes,16,(int)length,StandardCharsets.UTF_8)).getJSONArray("buffers");
     for(int i=0;i<buffers.length();i++){JSONObject b=buffers.getJSONObject(i),meta=desc.getJSONObject(i),v=b.getJSONArray("samples").getJSONObject(0);String hex=v.getString("hex");JSONArray offsets=v.getJSONArray("component_byte_offsets");int width=b.getInt("element_bytes");for(int j=0;j<offsets.length();j++)for(int k=0;k<width;k++)assertEquals(Integer.parseInt(hex.substring((j*width+k)*2,(j*width+k+1)*2),16),bytes[(int)(16+length+meta.getLong("payload_offset")+offsets.getLong(j)+k)]&255);}
    }finally{raw.delete();}
    assertEquals(data.getInt("step"),read(h,-1,-1).getInt("step"));assertEquals("OK",send(h,"Remember the code word ORCHID. Reply only OK."));assertEquals("ORCHID",send(h,"What code word did I ask you to remember? Reply only the word."));NativeModel.reset(h);assertEquals(0,read(h,-1,-1).getInt("step"));
   }finally{NativeModel.close(h);}
  }
 }
}
