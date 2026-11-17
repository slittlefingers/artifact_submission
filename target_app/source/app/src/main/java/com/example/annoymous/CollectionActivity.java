package com.example.annoymous;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.*;
import android.content.Intent;
import android.view.*;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.*;

                                                                                        
public final class CollectionActivity extends Activity {
 private final Handler handler=new Handler(Looper.getMainLooper());
 private final ExecutorService io=Executors.newSingleThreadExecutor();
 private TextView status,detail; private EditText name,context; private Spinner profiles;
 private Button pause,resume,change,create; private boolean active,busy; private String profileKey="";
 private final Runnable poll=new Runnable(){public void run(){request("status",null);if(active)handler.postDelayed(this,3000);}};
 private int dp(int n){return Math.round(n*getResources().getDisplayMetrics().density);}
 private Button button(LinearLayout parent,String title,Runnable action){Button b=new Button(this);b.setText(title);b.setAllCaps(false);b.setOnClickListener(v->action.run());parent.addView(b);return b;}
 @Override public void onCreate(Bundle state){
  super.onCreate(state);ScreenPrivacy.apply(this,getWindow());
  LinearLayout body=new LinearLayout(this);body.setOrientation(LinearLayout.VERTICAL);body.setPadding(dp(20),dp(18),dp(20),dp(24));body.setBackgroundColor(0xfff7f5f5);
  body.setOnApplyWindowInsetsListener((v,w)->{android.graphics.Insets i=w.getInsets(WindowInsets.Type.systemBars());v.setPadding(dp(20)+i.left,dp(12)+i.top,dp(20)+i.right,dp(24)+i.bottom);return w;});
  ScrollView scroll=new ScrollView(this);scroll.addView(body);setContentView(scroll);
  TextView title=new TextView(this);title.setText("Data collection");title.setTextSize(28);body.addView(title);
  status=new TextView(this);status.setTextSize(23);status.setText("Connecting to the control service...");body.addView(status);
  detail=new TextView(this);detail.setTextSize(15);body.addView(detail);
  pause=button(body,"Pause collection",()->request("pause",new JSONObject()));
  resume=button(body,"Start / resume",()->request("resume",new JSONObject()));
  TextView label=new TextView(this);label.setText("Context profile (pause before switching)");body.addView(label);
  profiles=new Spinner(this);body.addView(profiles);
  change=button(body,"Switch to selected profile",()->{try{request("profile",new JSONObject().put("name",String.valueOf(profiles.getSelectedItem())));}catch(Exception e){error(e);}});
  name=new EditText(this);name.setHint("New profile name, e.g. context_short");name.setSingleLine(true);body.addView(name);
  context=new EditText(this);context.setHint("Text before the image; for a full conversation, enter a messages JSON array\nExample: [{\"role\":\"user\",\"content\":\"Look at the image\"}]");context.setMinLines(4);context.setGravity(Gravity.TOP);body.addView(context);
  create=button(body,"Create context profile",()->{try{
   String text=context.getText().toString().trim();JSONArray messages;
   if(text.startsWith("["))messages=new JSONArray(text);
   else messages=new JSONArray().put(new JSONObject().put("role","user").put("content",text));
   request("create",new JSONObject().put("name",name.getText().toString().trim()).put("messages",messages));
  }catch(Exception e){error(e);}});
  TextView note=new TextView(this);note.setText("A 5-minute break is taken every 500 images. Each profile uses a separate data directory. Connect the phone by USB to the computer running the control service. Closing this screen does not stop collection.");body.addView(note);
  button(body,"Back to Ghost",()->{startActivity(new Intent(this,MainActivity.class));finish();});
 }
 private void error(Exception e){detail.setText("Control request failed: "+e.getMessage());}
 private JSONObject exchange(String endpoint,JSONObject payload)throws Exception{
  File file=new File(getExternalFilesDir(null),"collection_connection.json");
  JSONObject cfg=new JSONObject(new String(Files.readAllBytes(file.toPath()),StandardCharsets.UTF_8));
                                                                                 
  URL url=new URL("http://127.0.0.1:8765/api/"+endpoint);
  HttpURLConnection c=(HttpURLConnection)url.openConnection();c.setConnectTimeout(4000);c.setReadTimeout(10000);
  try{
   if(payload!=null){c.setRequestMethod("POST");c.setRequestProperty("X-Control-Token",cfg.getString("token"));c.setRequestProperty("Content-Type","application/json");c.setDoOutput(true);try(OutputStream out=c.getOutputStream()){out.write(payload.toString().getBytes(StandardCharsets.UTF_8));}}
   int code=c.getResponseCode();InputStream stream=code<400?c.getInputStream():c.getErrorStream();
   String text;try(InputStream in=stream){text=new String(in.readAllBytes(),StandardCharsets.UTF_8);}
   if(code>=400)throw new IOException(text);return new JSONObject(text);
  }finally{c.disconnect();}
 }
 private void request(String endpoint,JSONObject payload){
  if(busy)return;busy=true;
  io.execute(()->{try{JSONObject result=exchange(endpoint,payload);if(!endpoint.equals("status"))result=exchange("status",null);final JSONObject data=result;
   runOnUiThread(()->{busy=false;if(active)render(data);});
  }catch(Exception e){runOnUiThread(()->{busy=false;if(active){status.setText("Control service disconnected");error(e);}});}});
 }
 private void render(JSONObject d){
  String mode=d.optString("mode");status.setText(d.optString("profile")+"\n"+d.optInt("completed")+" / "+d.optInt("total"));
  JSONObject build=d.optJSONObject("build"),thermal=d.optJSONObject("thermal");
  String phase=mode.equals("paused")?"Paused":mode.equals("pausing")?"Saving and closing the worker":mode.equals("waiting")?"Waiting for the break to end":mode.equals("complete")?"Collection complete":mode.equals("running")?"Collecting / initializing":mode;
  if(build!=null&&mode.equals("running")&&build.optString("stage").startsWith("cooling"))phase="Cooling down";
  detail.setText(phase+" | Break remaining: "+d.optInt("cooldown")+" s\nThermal status: "+(thermal==null?"Unknown":thermal.optInt("thermal_status"))+ (d.isNull("error")?"":"\n"+d.optString("error")));
  pause.setEnabled((d.optBoolean("active")||mode.equals("waiting"))&&!mode.equals("pausing"));resume.setEnabled(!d.optBoolean("active")&&!mode.equals("waiting")&&!mode.equals("pausing")&&!mode.equals("complete"));change.setEnabled(!d.optBoolean("active")&&!mode.equals("pausing")&&!mode.equals("waiting"));
  JSONArray list=d.optJSONArray("profiles");String key=list==null?"":list.toString();if(!key.equals(profileKey)&&list!=null){profileKey=key;String[] names=new String[list.length()];int selected=0;for(int i=0;i<names.length;i++){names[i]=list.optJSONObject(i).optString("name");if(names[i].equals(d.optString("profile")))selected=i;}profiles.setAdapter(new ArrayAdapter<String>(this,android.R.layout.simple_spinner_dropdown_item,names));profiles.setSelection(selected);}
 }
 @Override protected void onResume(){super.onResume();active=true;handler.post(poll);}
 @Override protected void onPause(){active=false;handler.removeCallbacks(poll);super.onPause();}
 @Override protected void onDestroy(){io.shutdownNow();super.onDestroy();}
}
