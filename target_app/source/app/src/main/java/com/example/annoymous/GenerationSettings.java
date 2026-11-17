package com.example.annoymous;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.text.InputType;
import android.widget.*;

                                                                                  
final class GenerationSettings {
 final float temperature, topP;
 final int topK, maxOutputTokens;
 GenerationSettings(float temperature,int topK,float topP,int maxOutputTokens){
  if(!Float.isFinite(temperature)||temperature<0||temperature>2)throw new IllegalArgumentException("Temperature must be between 0 and 2.");
  if(topK<1||topK>1024)throw new IllegalArgumentException("Top-K must be between 1 and 1024.");
  if(!Float.isFinite(topP)||topP<=0||topP>1)throw new IllegalArgumentException("Top-P must be greater than 0 and at most 1.");
  if(maxOutputTokens<1||maxOutputTokens>4096)throw new IllegalArgumentException("Max output tokens must be between 1 and 4096.");
  this.temperature=temperature;this.topK=topK;this.topP=topP;this.maxOutputTokens=maxOutputTokens;
 }
 static GenerationSettings defaults(){return new GenerationSettings(1f,1,1f,512);}
 static GenerationSettings load(Context c){
  SharedPreferences p=c.getSharedPreferences("generation_settings",Context.MODE_PRIVATE);
  try{return new GenerationSettings(p.getFloat("temperature",1f),p.getInt("top_k",1),p.getFloat("top_p",1f),p.getInt("max_output_tokens",512));}
  catch(IllegalArgumentException e){return defaults();}
 }
 void save(Context c){c.getSharedPreferences("generation_settings",Context.MODE_PRIVATE).edit().putFloat("temperature",temperature).putInt("top_k",topK).putFloat("top_p",topP).putInt("max_output_tokens",maxOutputTokens).apply();}
 void startSession(long handle){NativeModel.resetWithGenerationSettings(handle,temperature,topK,topP,maxOutputTokens);}
 private static EditText field(Activity a,LinearLayout panel,String label,String value,boolean decimal){
  TextView title=new TextView(a);title.setText(label);title.setPadding(0,16,0,0);panel.addView(title);
  EditText input=new EditText(a);input.setSingleLine(true);input.setContentDescription(label);
  input.setInputType(InputType.TYPE_CLASS_NUMBER|(decimal?InputType.TYPE_NUMBER_FLAG_DECIMAL:0));input.setText(value);panel.addView(input);return input;
 }
 static void show(Activity a){
  GenerationSettings current=load(a);
  LinearLayout panel=new LinearLayout(a);panel.setOrientation(LinearLayout.VERTICAL);
  int pad=(int)(24*a.getResources().getDisplayMetrics().density);panel.setPadding(pad,0,pad,pad);
  TextView note=new TextView(a);note.setText("Applies to all chat models from the next new session or model load. Saving keeps the current conversation.\n\nTemperature 0 or Top-K 1 uses greedy decoding. Responses may end earlier at the model's stop token or context limit.");panel.addView(note);
  EditText temp=field(a,panel,"Temperature (0–2)",Float.toString(current.temperature),true);
  EditText k=field(a,panel,"Top-K (1–1024)",Integer.toString(current.topK),false);
  EditText p=field(a,panel,"Top-P (0–1, excluding 0)",Float.toString(current.topP),true);
  EditText max=field(a,panel,"Max output tokens (1–4096)",Integer.toString(current.maxOutputTokens),false);
  TextView error=new TextView(a);error.setTextColor(0xffb3261e);panel.addView(error);
  ScrollView scroll=new ScrollView(a);scroll.addView(panel);
  AlertDialog dialog=new AlertDialog.Builder(a).setTitle("Generation settings").setView(scroll).setPositiveButton("Save",null).setNegativeButton("Cancel",null).setNeutralButton("Defaults",null).create();
  ScreenPrivacy.show(a,dialog);
  dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener(v->{GenerationSettings d=defaults();temp.setText(Float.toString(d.temperature));k.setText(Integer.toString(d.topK));p.setText(Float.toString(d.topP));max.setText(Integer.toString(d.maxOutputTokens));error.setText("");});
  dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v->{
   try{new GenerationSettings(Float.parseFloat(temp.getText().toString().trim()),Integer.parseInt(k.getText().toString().trim()),Float.parseFloat(p.getText().toString().trim()),Integer.parseInt(max.getText().toString().trim())).save(a);dialog.dismiss();Toast.makeText(a,"Saved. Applies to the next new session.",Toast.LENGTH_LONG).show();}
   catch(NumberFormatException e){error.setText("Enter a valid number in every field.");}
   catch(IllegalArgumentException e){error.setText(e.getMessage());}
  });
 }
}
