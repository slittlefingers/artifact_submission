package com.example.annoymous;
import android.app.*;
import android.content.*;
import android.os.SystemClock;
import android.widget.*;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.lang.reflect.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;
@RunWith(AndroidJUnit4.class)
public class ModelCatalogUiDeviceTest {
 private Object get(MainActivity a,String name)throws Exception{Field f=MainActivity.class.getDeclaredField(name);f.setAccessible(true);return f.get(a);}
 @Test public void importedModelsAndImageControls()throws Exception{
  Instrumentation inst=InstrumentationRegistry.getInstrumentation();Context c=inst.getTargetContext();MainActivity a=(MainActivity)inst.startActivitySync(new Intent(c,MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
  try{
   long end=SystemClock.uptimeMillis()+120000;while(SystemClock.uptimeMillis()<end&&((Boolean)get(a,"busy")||(Long)get(a,"handle")==0))SystemClock.sleep(200);
   assertTrue("Qwen model should load",(Long)get(a,"handle")!=0);assertEquals(4,((String[])get(a,"MODEL_FILES")).length);
   Method ready=MainActivity.class.getDeclaredMethod("modelReady",int.class);ready.setAccessible(true);for(int i=0;i<4;i++)assertTrue("Imported model "+i,(Boolean)ready.invoke(a,i));
   Field selected=MainActivity.class.getDeclaredField("selectedModel");selected.setAccessible(true);int before=selected.getInt(a);Method setBusy=MainActivity.class.getDeclaredMethod("setBusy",boolean.class,String.class);setBusy.setAccessible(true);
   inst.runOnMainSync(()->{try{for(int i=0;i<4;i++){selected.setInt(a,i);setBusy.invoke(a,false,"QA");assertEquals(i==0||i==1||i==3,((ImageButton)get(a,"load")).isEnabled());assertEquals(i==0,((ImageButton)get(a,"microphone")).isEnabled());}selected.setInt(a,before);setBusy.invoke(a,false,"Ready");}catch(Exception e){throw new RuntimeException(e);}});
  }finally{inst.runOnMainSync(()->a.finish());}
 }
}
