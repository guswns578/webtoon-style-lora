"""Webtoon Character Studio: local-only generation, persistent serial queue and review assets."""
from __future__ import annotations
import base64,copy,hashlib,io,json,os,re,sqlite3,threading,time,urllib.request,uuid
from pathlib import Path
from contextlib import contextmanager
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFilter,ImageOps,PngImagePlugin

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime';RESULTS=ROOT/'results'
for p in [RUNTIME,RESULTS]:p.mkdir(exist_ok=True)
PRESETS=json.loads((ROOT/'presets.json').read_text(encoding='utf-8'))
DB=RUNTIME/'jobs.sqlite3';LOCK=threading.RLock();SESSION=None;ACTIVE=None
@contextmanager
def connect_db():
 connection=sqlite3.connect(DB)
 try:
  with connection:yield connection
 finally:connection.close()

def dump(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)
def settings():
 p=RUNTIME/'settings.json'
 return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'api_url':'http://127.0.0.1:7861','forge_dir':'','models_dir':str(ROOT/'models')}
def save_settings(url,forge,models):
 url=url.strip().rstrip('/')
 if not re.fullmatch(r'http://(?:127\.0\.0\.1|localhost):\d{2,5}',url):raise ValueError('연결 주소는 http://127.0.0.1:포트 형식으로 입력하세요.')
 dump(RUNTIME/'settings.json',{'api_url':url,'forge_dir':forge.strip(),'models_dir':models.strip() or str(ROOT/'models')})
def api(route,payload=None,timeout=900):
 req=urllib.request.Request(settings()['api_url']+route,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
 except urllib.error.HTTPError as e:
  detail=e.read().decode('utf-8',errors='replace')[:1500];raise RuntimeError(f'Forge 응답 오류 {e.code}: {detail}') from e
 except urllib.error.URLError as e:raise RuntimeError('Forge에 연결할 수 없습니다. 환경 점검에서 Forge 실행 상태와 주소를 확인하세요.') from e
def b64(im):
 b=io.BytesIO();im.save(b,format='PNG');return base64.b64encode(b.getvalue()).decode()
def decode(s):return Image.open(io.BytesIO(base64.b64decode(s.split(',')[-1]))).convert('RGB')
def levels(im,lo=35,hi=210):
 if not 0<=lo<hi<=255:raise ValueError('선화 보정의 검정 기준은 흰색 기준보다 작아야 합니다.')
 a=np.asarray(im.convert('L'),dtype='float32');return Image.fromarray(np.clip((a-lo)*255/(hi-lo),0,255).astype('uint8')).convert('RGB')
def isolate(raw,lo=35,hi=210):
 out=levels(raw,lo,hi).convert('RGBA');g=np.asarray(out.convert('L'));ink=(g<245).astype('uint8')
 closed=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
 _,lab=cv2.connectedComponents(1-closed,8);border=set(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]).tolist());border.discard(0)
 filled=(~np.isin(lab,list(border))).astype('uint8');n,parts,stats,_=cv2.connectedComponentsWithStats(filled,8)
 if n<2:raise ValueError('인물 영역을 분리하지 못했습니다. 생성 원본을 확인하고 분리 마스크를 수정하세요.')
 k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));out.putalpha(Image.fromarray((parts==k).astype('uint8')*255));return out
def face_session():
 global SESSION
 if SESSION is None:
  import onnxruntime as ort
  opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
  SESSION=ort.InferenceSession(str(ROOT/'assets'/'face_detector.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
 return SESSION
def detect_face(raw):
 src=raw.convert('RGB');side=1280;ratio=min(side/src.width,side/src.height);w,h=round(src.width*ratio),round(src.height*ratio);x,y=(side-w)//2,(side-h)//2
 canvas=Image.new('RGB',(side,side),'white');canvas.paste(src.resize((w,h)),(x,y));tensor=np.asarray(canvas,dtype='float32').transpose(2,0,1)[None]/255
 sess=face_session();pred=sess.run(None,{sess.get_inputs()[0].name:tensor})[0][0]
 if pred.shape[0]<pred.shape[1]:pred=pred.T
 faces=[]
 for row in pred:
  if row[4]<.6:continue
  cx,cy,bw,bh=map(float,row[:4]);box=[max(0,(cx-bw/2-x)/ratio),max(0,(cy-bh/2-y)/ratio),min(src.width,(cx+bw/2-x)/ratio),min(src.height,(cy+bh/2-y)/ratio)]
  if box[3]<src.height*.6 and box[2]-box[0]>=12:faces.append({'box':box,'score':float(row[4])})
 return max(faces,key=lambda f:f['score']) if faces else None
def face_mask(raw,alias):
 mask=Image.new('L',raw.size,0);face=detect_face(raw)
 if not face:return mask,None
 x,y,x2,y2=face['box'];w,h=x2-x,y2-y
 eye=(.42,.62) if alias=='char_h' else (.27,.46);lower=(.64,.85) if alias=='char_h' else (.49,.83)
 if alias=='char_c':rects=[(x+w*.60,y+h*eye[0],x+w*.90,y+h*eye[1]),(x+w*.51,y+h*.51,x+w*.78,y+h*.82)]
 else:rects=[(x+w*.13,y+h*eye[0],x+w*.40,y+h*eye[1]),(x+w*.60,y+h*eye[0],x+w*.87,y+h*eye[1]),(x+w*.31,y+h*lower[0],x+w*.73,y+h*lower[1])]
 d=ImageDraw.Draw(mask)
 for box in rects:d.rectangle(tuple(round(v) for v in box),fill=255)
 return mask,face
def pose_image(alias,choice):
 char=PRESETS['characters'][alias];spec=PRESETS['poses'][char['recommended_pose'] if choice=='recommended' else choice]
 points=[list(p) for p in PRESETS['standing_points']]
 for k,v in spec.get('points',{}).items():points[int(k)]=v
 if choice not in ['recommended',char['recommended_pose']]:
  original_width=abs(points[5][0]-points[2][0]);factor=(236*char['spread'])/max(1,original_width)
  points=[(round(448+(x-448)*factor),y) for x,y in points]
 image=Image.new('RGB',(896,1152),'black');d=ImageDraw.Draw(image)
 edges=[(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(1,0)]
 colors=[(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255)]
 for (a,b),c in zip(edges,colors):d.line((points[a],points[b]),fill=c,width=7)
 for i,(x,y) in enumerate(points):d.ellipse((x-5,y-5,x+5,y+5),fill=colors[i%len(colors)])
 return image,spec['prompt']
def background(params):
 if params.get('_preview_background') is not None:
  return ImageOps.pad(params['_preview_background'].convert('RGB'),(1024,1280),color='white')
 if params.get('background_path'):
  return ImageOps.pad(Image.open(params['background_path']).convert('RGB'),(1024,1280),color='white')
 im=Image.new('RGB',(1024,1280),'white')
 if params.get('background_mode','white')=='white':return im
 d=ImageDraw.Draw(im)
 for y in [240,445,495,565,665,805,1010,1250]:d.line((0,y,1024,y),fill=(180,180,180),width=2)
 for x in [-1500,-700,0,700,1400,2200]:d.line((700,445,x,1280),fill=(175,175,175),width=2)
 for x in [80,300,520,740,960]:d.line((x,240,x,445),fill=(190,190,190),width=2)
 return im
def render(rgba,shot,params):
 bounds=rgba.getchannel('A').getbbox()
 if not bounds:raise ValueError('인물 마스크가 비어 있습니다. 흰색으로 인물 영역을 표시하세요.')
 cut=rgba.crop(bounds);height=round(1280*(params.get('b_height',125) if shot=='B' else params.get('c_height',33.6))/100)
 size=(max(1,round(cut.width*height/cut.height)),height)
 if size[0]>1004:
  scale=1004/size[0];size=(1004,max(1,round(size[1]*scale)))
 cut=cut.resize(size,Image.Resampling.LANCZOS)
 if shot=='C':
  a=np.asarray(cut).copy();g=cv2.cvtColor(a[:,:,:3],cv2.COLOR_RGB2GRAY);med=cv2.medianBlur(g,5);inside=cv2.distanceTransform((a[:,:,3]>128).astype('uint8'),cv2.DIST_L2,3)>3
  high=((med.astype('int16')-g.astype('int16')>22)&inside).astype('uint8');n,lab,stats,_=cv2.connectedComponentsWithStats(high,8);remove=np.zeros(g.shape,dtype=bool)
  for k in range(1,n):
   if stats[k,cv2.CC_STAT_AREA]<=10:remove|=lab==k
  a[remove,:3]=np.repeat(med[remove,None],3,axis=1);cut=Image.fromarray(a)
 cx=.635 if shot=='B' else .742;y=180 if shot=='B' else 490
 pos=(max(10,min(1024-cut.width-10,round(1024*cx-cut.width/2))),y)
 panel=background(params).convert('RGBA');panel.alpha_composite(cut,pos)
 if params.get('foreground',False):
  fg=Image.open(params.get('foreground_path') or ROOT/'assets'/'foreground.png').convert('RGBA');fg=fg.resize((round(fg.width*900/fg.height),900),Image.Resampling.LANCZOS);panel.alpha_composite(fg,(-max(80,fg.width//4),670))
 ImageDraw.Draw(panel).rectangle((4,4,1019,1275),outline='black',width=4)
 return panel.convert('RGB'),cut,{'position':list(pos),'cutout_size':list(cut.size),'height_fraction':cut.height/1280,'outline_protection_px':3 if shot=='C' else None}
def compose_files(raw,params,folder,figure_mask=None,omission_mask=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);raw=raw.convert('RGB');warnings=[]
 if figure_mask is None:isolated=isolate(raw,params.get('lo',35),params.get('hi',210))
 else:
  isolated=levels(raw,params.get('lo',35),params.get('hi',210)).convert('RGBA');isolated.putalpha(figure_mask.convert('L').resize(raw.size))
 if omission_mask is None:
  try:omission_mask,face=face_mask(raw,params['character'])
  except Exception as e:omission_mask=Image.new('L',raw.size,0);face=None;warnings.append('얼굴 검출 실패: '+str(e))
  if face is None:warnings.append('얼굴을 찾지 못했습니다. C컷 생략 마스크를 직접 지정한 뒤 다시 합성하세요.')
 else:face=None
 omission_mask=omission_mask.convert('L').resize(raw.size)
 simple=Image.composite(Image.new('RGBA',raw.size,'white'),isolated,omission_mask);simple.putalpha(isolated.getchannel('A'))
 alpha=np.asarray(isolated.getchannel('A'))
 if np.any(alpha[0]>128) or np.any(alpha[-1]>128):warnings.append('인물이 원본의 위·아래 경계에 닿습니다. 머리·발·무기 끝이 잘리지 않았는지 확인하세요.')
 isolated.save(folder/'figure.png');isolated.getchannel('A').save(folder/'figure_mask.png');omission_mask.save(folder/'omission_mask.png')
 result={'raw':str(folder/'raw.png'),'figure':str(folder/'figure.png'),'figure_mask':str(folder/'figure_mask.png'),'omission_mask':str(folder/'omission_mask.png'),'warnings':warnings,'face':face,'layout':{}}
 raw.save(folder/'raw.png')
 for shot in ['B','C']:
  if params.get('shots','both') not in ['both',shot]:continue
  panel,cut,layout=render(isolated if shot=='B' else simple,shot,params);panel.save(folder/f'{shot}.png');cut.save(folder/f'{shot}_cutout.png');result[shot]=str(folder/f'{shot}.png');result[shot+'_cutout']=str(folder/f'{shot}_cutout.png');result['layout'][shot]=layout
 dump(folder/'result.json',result);dump(folder/'params.json',params);return result
def init_db():
 with connect_db() as c:
  c.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,created TEXT,status TEXT,params TEXT,result TEXT,error TEXT,stage TEXT,cancel INTEGER DEFAULT 0)')
  c.execute("UPDATE jobs SET status='failed',error='도구가 종료되어 작업이 중단됐습니다. 재시도할 수 있습니다.' WHERE status='running'")
def job(id):
 with connect_db() as c:
  c.row_factory=sqlite3.Row;r=c.execute('SELECT * FROM jobs WHERE id=?',(id,)).fetchone()
 return dict(r) if r else None
def jobs():
 with connect_db() as c:
  c.row_factory=sqlite3.Row;return [dict(x) for x in c.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 200')]
def update(id,**fields):
 with LOCK,connect_db() as c:c.execute('UPDATE jobs SET '+','.join(k+'=?' for k in fields)+' WHERE id=?',list(fields.values())+[id])
def enqueue(params):
 p=copy.deepcopy(params);id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:7]
 if int(p['seed'])<0:p['seed']=int.from_bytes(os.urandom(4),'little')%(2**31)
 p['seed']=int(p['seed'])
 with LOCK,connect_db() as c:c.execute('INSERT INTO jobs (id,created,status,params,stage) VALUES (?,?,?,?,?)',(id,time.strftime('%Y-%m-%d %H:%M:%S'),'queued',json.dumps(p,ensure_ascii=False),'대기'))
 return id
def cancelled(id):return bool(job(id)['cancel'])
def run_job(id,params):
 char=PRESETS['characters'][params['character']];folder=RESULTS/id;folder.mkdir(exist_ok=True)
 update(id,stage='포즈 준비')
 if params.get('pose_path'):
  source=Image.open(params['pose_path']).convert('RGB');source=ImageOps.pad(source,(896,1152),color='black' if params['pose_kind']=='openpose' else 'white')
  source.save(folder/'pose_input.png')
  if params['pose_kind']=='openpose':pose=source
  else:
   detected=api('/controlnet/detect',{'controlnet_module':'dw_openpose_full','controlnet_input_images':[b64(source)],'controlnet_processor_res':1024})
   if not detected.get('images'):raise RuntimeError('포즈를 추출하지 못했습니다. 다른 이미지 또는 OpenPose 이미지를 사용하세요.')
   pose=decode(detected['images'][0])
  pose_text=params.get('pose_description','standing in the reference pose')
 else:pose,pose_text=pose_image(params['character'],params.get('pose','recommended'))
 pose.save(folder/'pose.png')
 if cancelled(id):raise InterruptedError('취소 요청에 따라 중지했습니다.')
 prompt=re.sub(r'<lora:[^>]+>',f"<lora:{char['lora']}:{params.get('weight',.9):.2f}>",char['prompt'])+', '+pose_text
 if params.get('extra'):prompt+=', '+params['extra'].strip()
 models=api('/sdapi/v1/sd-models',timeout=15);matched=[m for m in models if 'waiIllustriousSDXL_v160' in m.get('model_name','') or 'waiIllustriousSDXL_v160' in m.get('title','')]
 if not matched:raise RuntimeError('Forge에서 waiIllustriousSDXL_v160 체크포인트를 찾지 못했습니다. 모델 경로를 확인하세요.')
 control={'image':b64(pose),'module':'None','model':'thibaud_xl_openpose','weight':.95,'pixel_perfect':False,'processor_res':1024,'resize_mode':'Just Resize','control_mode':'Balanced','type_filter':'OpenPose','enabled':True,'guidance_start':0.,'guidance_end':1.,'save_detected_map':False}
 p={'prompt':prompt,'negative_prompt':char['negative'],'seed':params['seed'],'steps':28,'cfg_scale':5.,'width':896,'height':1152,'sampler_name':'Euler a','scheduler':'Automatic','save_images':False,'do_not_save_samples':True,'do_not_save_grid':True,'override_settings':{'sd_model_checkpoint':matched[0]['title'],'CLIP_stop_at_last_layers':1,'randn_source':'CPU'},'alwayson_scripts':{'ControlNet':{'args':[control]}}}
 clean=copy.deepcopy(p);clean['alwayson_scripts']['ControlNet']['args'][0].pop('image');clean['pose_sha256']=hashlib.sha256((folder/'pose.png').read_bytes()).hexdigest();dump(folder/'request.json',clean)
 update(id,stage='캐릭터 생성 중');response=api('/sdapi/v1/txt2img',p);info=json.loads(response['info']);dump(folder/'generation.json',info)
 if info.get('sd_model_hash')!='a5f58eb1c3':raise RuntimeError('생성된 체크포인트 해시가 검증 기준과 다릅니다. 환경 점검에서 모델을 확인하세요.')
 applied=str(info.get('extra_generation_params',{}).get('Lora hashes',''))
 if char['lora']+': '+char['forge_hash'] not in applied:raise RuntimeError('생성 기록에서 선택한 LoRA 적용을 확인하지 못했습니다. Forge 모델 등록과 로그를 확인하세요.')
 raw=decode(response['images'][0]);meta=PngImagePlugin.PngInfo();meta.add_text('parameters',(info.get('infotexts') or [''])[0]);raw.save(folder/'generated_raw.png',pnginfo=meta)
 if cancelled(id):raise InterruptedError('생성은 저장됐으며, 취소 요청에 따라 후속 합성은 중지했습니다.')
 update(id,stage='인물 분리·B/C 합성');result=compose_files(raw,params,folder)
 result['generated_raw']=str(folder/'generated_raw.png');result['generation']=str(folder/'generation.json');dump(folder/'result.json',result)
 return result
def worker():
 global ACTIVE
 while True:
  with connect_db() as c:
   c.row_factory=sqlite3.Row;row=c.execute("SELECT * FROM jobs WHERE status='queued' AND cancel=0 ORDER BY created,rowid LIMIT 1").fetchone()
  if row is None:time.sleep(.6);continue
  j=dict(row);ACTIVE=j['id'];update(j['id'],status='running',stage='준비')
  try:
   result=run_job(j['id'],json.loads(j['params']));update(j['id'],status='review',result=json.dumps(result,ensure_ascii=False),stage='검수 대기',error='')
  except InterruptedError as e:update(j['id'],status='cancelled',error=str(e),stage='취소됨')
  except Exception as e:
   import traceback
   folder=RESULTS/j['id'];folder.mkdir(exist_ok=True);(folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8');update(j['id'],status='failed',error=str(e),stage='실패')
  finally:ACTIVE=None
def start_worker():
 init_db();threading.Thread(target=worker,daemon=True,name='serial-worker').start()
def preflight():
 cfg=settings();base=Path(cfg['models_dir']);rows=[]
 files=[('베이스 모델','Stable-diffusion/waiIllustriousSDXL_v160.safetensors'),('자세 제어 모델','ControlNet/thibaud_xl_openpose.safetensors')]+[(v['name']+' LoRA','Lora/'+v['lora']+'.safetensors') for v in PRESETS['characters'].values()]
 for name,path in files:
  p=base/path;rows.append([name,'확인' if p.is_file() and p.stat().st_size>1000 else '없음',str(p)])
 for label,path in [('얼굴 검출',ROOT/'assets'/'face_detector.onnx'),('전경 이미지',ROOT/'assets'/'foreground.png')]:rows.append([label,'확인' if path.is_file() else '없음',str(path)])
 try:
  opts=api('/sdapi/v1/options',timeout=5);cn=api('/controlnet/model_list',timeout=5);rows.append(['Forge 연결','확인',str(opts.get('sd_model_checkpoint',''))])
  rows.append(['OpenPose Forge 등록','확인' if any('thibaud_xl_openpose' in x for x in cn.get('model_list',[])) else '없음','thibaud_xl_openpose'])
  loras=api('/sdapi/v1/loras',timeout=5);available={x['name'] for x in loras}
  for v in PRESETS['characters'].values():
   if v['lora'] not in available:rows.append([v['name']+' Forge 등록','없음','모델 폴더를 Forge에 연결한 뒤 다시 시작하세요.'])
 except Exception as e:rows.append(['Forge 연결','미연결',str(e)])
 return rows
