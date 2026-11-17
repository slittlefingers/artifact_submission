package com.example.annoymous;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import java.io.File;

                                                                                   
public final class LauncherActivity extends Activity {
 @Override public void onCreate(Bundle state){
  super.onCreate(state);
  startActivity(new Intent(this,new File(getFilesDir(),"attack_live.json").isFile()?AttackActivity.class:MainActivity.class));
  finish();
 }
}
