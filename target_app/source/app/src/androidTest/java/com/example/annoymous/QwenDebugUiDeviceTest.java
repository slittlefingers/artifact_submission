package com.example.annoymous;
import android.app.*;
import android.content.*;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.view.*;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.*;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.*;
import java.lang.reflect.*;
import java.nio.file.Files;
import static org.junit.Assert.*;
@RunWith(AndroidJUnit4.class)
public class QwenDebugUiDeviceTest {
 private Instrumentation inst;private MainActivity activity;
 private Object field(String name)throws Exception{Field f=MainActivity.class.getDeclaredField(name);f.setAccessible(true);return f.get(activity);}
 private void invoke(String name)throws Exception{Method m=MainActivity.class.getDeclaredMethod(name);m.setAccessible(true);inst.runOnMainSync(()->{try{m.invoke(activity);}catch(Exception e){throw new RuntimeException(e);}});}
 private AccessibilityNodeInfo node(String label){AccessibilityNodeInfo root=inst.getUiAutomation().getRootInActiveWindow();if(root==null)return null;for(AccessibilityNodeInfo n:root.findAccessibilityNodeInfosByText(label)){if(n.getText()!=null&&n.getText().toString().equals(label))return n;}return null;}
 private void waitText(String label){long end=SystemClock.uptimeMillis()+30000;while(SystemClock.uptimeMillis()<end){if(node(label)!=null)return;SystemClock.sleep(150);}fail("UI text not found: "+label);}
 private void click(String label){waitText(label);AccessibilityNodeInfo n=node(label);while(n!=null&&!n.isClickable())n=n.getParent();assertNotNull("Clickable "+label,n);assertTrue(n.performAction(AccessibilityNodeInfo.ACTION_CLICK));SystemClock.sleep(250);}
 private boolean scroll(AccessibilityNodeInfo n){if(n==null)return false;if(n.isScrollable()&&n.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD))return true;for(int i=0;i<n.getChildCount();i++)if(scroll(n.getChild(i)))return true;return false;}
 private void layer(String label){for(int i=0;i<12&&node(label)==null;i++){scroll(inst.getUiAutomation().getRootInActiveWindow());SystemClock.sleep(350);}click(label);}
 private void ready()throws Exception{long end=SystemClock.uptimeMillis()+90000;while(SystemClock.uptimeMillis()<end){if(!((Boolean)field("busy"))&&((Long)field("handle"))!=0)return;SystemClock.sleep(200);}fail("Model never ready: "+((TextView)field("status")).getText());}
 private void screenshot(String name)throws Exception{SystemClock.sleep(400);Bitmap b=inst.getUiAutomation().takeScreenshot();assertNotNull(b);try(FileOutputStream out=new FileOutputStream(new File(activity.getCacheDir(),name))){assertTrue(b.compress(Bitmap.CompressFormat.PNG,100,out));}b.recycle();}
 @Test public void twoLongUiTurns()throws Exception{
  inst=InstrumentationRegistry.getInstrumentation();Context c=inst.getTargetContext();
  Intent intent=new Intent(c,MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
  activity=(MainActivity)inst.startActivitySync(intent);
  try{
   ready();assertEquals(2,((Integer)field("selectedModel")).intValue());
   for(String prompt:new String[]{"Explain how rain forms. Give a detailed answer in several sentences.","Continue with a detailed explanation of how snow forms."}){
    inst.runOnMainSync(()->{try{((EditText)field("input")).setText(prompt);((ImageButton)field("send")).performClick();}catch(Exception e){throw new RuntimeException(e);}});
    ready();assertTrue(((TextView)field("transcript")).getText().length()>80);
   }
  }finally{if(activity!=null)inst.runOnMainSync(()->activity.finish());}
 }
 @Test public void linearAndGlobalViews()throws Exception{
  inst=InstrumentationRegistry.getInstrumentation();Context c=inst.getTargetContext();File privacy=new File(c.getFilesDir(),"screen_privacy");byte[] before=privacy.isFile()?Files.readAllBytes(privacy.toPath()):null;
                                                                                            
  Files.write(privacy.toPath(),new byte[]{'0'});
  try{
   Intent intent=new Intent(c,MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);activity=(MainActivity)inst.startActivitySync(intent);
   inst.runOnMainSync(()->{activity.setShowWhenLocked(true);activity.setTurnScreenOn(true);});ready();
   assertEquals(2,((Integer)field("selectedModel")).intValue());
   inst.runOnMainSync(()->{try{((EditText)field("input")).setText("What is 2 + 3? Reply with the number only.");((ImageButton)field("send")).performClick();}catch(Exception e){throw new RuntimeException(e);}});ready();assertEquals("5",((TextView)field("transcript")).getText().toString().trim());
   invoke("showSessionTools");click("Debug");assertTrue((Boolean)field("debugEnabled"));click("KV Cache");click("Layer 00  \u203a");waitText("R");waitText("C");click("R");waitText("head : row");screenshot("qwen-r-grid.png");click("0:0");waitText("R / head 0 / row 0");screenshot("qwen-r-values.png");inst.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK);click("C");click("0");waitText("C / channel 0");screenshot("qwen-c-values.png");inst.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK);SystemClock.sleep(500);inst.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK);SystemClock.sleep(500);waitText("KV Cache");layer("Layer 03  \u203a");waitText("K_3");waitText("V_3");screenshot("qwen-global-kv.png");
  }finally{
   if(before==null)privacy.delete();else Files.write(privacy.toPath(),before);
   if(activity!=null)inst.runOnMainSync(()->{ScreenPrivacy.refresh(activity);activity.finish();});
  }
 }
}
