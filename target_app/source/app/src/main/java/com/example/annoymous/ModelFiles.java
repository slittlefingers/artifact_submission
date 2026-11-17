package com.example.annoymous;
import android.content.Context;
import java.io.File;
final class ModelFiles {
 static File resolve(Context context,String name){
  File privateModel=new File(context.getFilesDir(),name);
  return privateModel.isFile()?privateModel:new File(context.getExternalFilesDir(null),name);
 }
}
