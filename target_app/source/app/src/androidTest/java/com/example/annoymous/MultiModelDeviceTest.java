package com.example.annoymous;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import static org.junit.Assert.*;
@RunWith(AndroidJUnit4.class)
public class MultiModelDeviceTest {
 private static final String[] MODELS={"llama3_2_1b_Google_Tensor_G5.litertlm","qwen3_5_2b_vl_Google_Tensor_G5.litertlm","ministral3_3b_vl_Google_Tensor_G5.litertlm","phi4_mini_Google_Tensor_G5.litertlm"};
 private String send(long h,String text,String image){String answer=new String(NativeModel.sendMedia(h,text.getBytes(StandardCharsets.UTF_8),image,""),StandardCharsets.UTF_8).trim();Log.i("MultiModelQA","reply="+answer);return answer;}
 private JSONObject inspect(long h)throws Exception{return new JSONObject(new String(NativeModel.inspect(h,"{}"),StandardCharsets.UTF_8));}
 private void textModel(int index)throws Exception{
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();File f=ModelFiles.resolve(c,MODELS[index]);assertTrue(f.toString(),f.isFile());long h=NativeModel.open(f.getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir);
  try{assertTrue(send(h,"What is 2 + 3? Reply with the number only.","").contains("5"));JSONObject state=inspect(h);assertTrue(state.getJSONArray("layers").length()>0);
   JSONArray tokens=state.getJSONArray("tokens"),allocations=state.getJSONArray("token_allocations");
   for(int i=0;i<tokens.length();i++){JSONObject t=tokens.getJSONObject(i);if(t.getString("region").equals("live")&&t.getInt("index")<allocations.getJSONObject(t.getInt("candidate")).getInt("size")-1&&t.getInt("id")>0)fail("Unscrubbed text ID in "+MODELS[index]);}
   assertTrue(send(h,"Remember the word ORCHID. Reply only OK.","").contains("OK"));assertTrue(send(h,"What word did I ask you to remember? Reply only the word.","").contains("ORCHID"));
  }finally{NativeModel.close(h);}
 }
 private void imageModel(int index)throws Exception{
  Context c=InstrumentationRegistry.getInstrumentation().getTargetContext();long h=NativeModel.open(ModelFiles.resolve(c,MODELS[index]).getAbsolutePath(),c.getCacheDir().getAbsolutePath(),c.getApplicationInfo().nativeLibraryDir);
  File image=new File(c.getCacheDir(),"qa-visual-input.png");
  try{for(int color:new int[]{Color.RED,Color.BLUE}){
   NativeModel.reset(h);Bitmap bitmap=Bitmap.createBitmap(448,448,Bitmap.Config.ARGB_8888);bitmap.eraseColor(color);try(FileOutputStream out=new FileOutputStream(image)){bitmap.compress(Bitmap.CompressFormat.PNG,100,out);}bitmap.recycle();
   String expected=color==Color.RED?"red":"blue";String answer=send(h,"Name the dominant color in this image. Reply with one word.",image.getAbsolutePath()).toLowerCase();assertTrue("Expected "+expected+", got "+answer,answer.contains(expected));
   assertTrue("Image token positions missing",inspect(h).getInt("step")>190);
   assertTrue(send(h,"What color did I just show you? Reply with one word.","").toLowerCase().contains(expected));
  }}finally{image.delete();NativeModel.close(h);}
 }
 @Test public void llamaText()throws Exception{textModel(0);}
 @Test public void qwenText()throws Exception{textModel(1);}
 @Test public void ministralText()throws Exception{textModel(2);}
 @Test public void phiText()throws Exception{textModel(3);}
 @Test public void qwenImages()throws Exception{imageModel(1);}
 @Test public void ministralImages()throws Exception{imageModel(2);}
}
