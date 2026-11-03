from __future__ import annotations
import json,os,shutil,sys,time,uuid,zipfile
from pathlib import Path
import gradio as gr
import numpy as np
from PIL import Image
import engine as e
import forge_control
from ui_helpers import editor_value,editor_mask,layout_preview,seed_value,progress_text

STATUS={'queued':'대기','running':'생성 중','review':'검수 대기','approved':'승인','rejected':'재작업','failed':'실패','cancelled':'취소'}
CHARS=[(v['name'],k) for k,v in e.PRESETS['characters'].items()]
POSES=[('캐릭터별 추천 포즈','recommended')]+[(v['label'],k) for k,v in e.PRESETS['poses'].items()]
def refresh(current=None):
 rows=e.jobs();table=[];choices=[]
 for j in rows:
  p=json.loads(j['params']);name=e.PRESETS['characters'][p['character']]['name'];status=STATUS.get(j['status'],j['status'])
  table.append([j['id'],name,status,j['stage'] or '',str(p['seed']),j['error'] or ''])
  choices.append((f"{name} · {status} · {j['id']}",j['id']))
 value=current if any(j['id']==current for j in rows) else (rows[0]['id'] if rows else None)
 return table,gr.update(choices=choices,value=value)
def copy_upload(image,label):
 if image is None:return None
 folder=e.RUNTIME/'inputs';folder.mkdir(exist_ok=True);p=folder/(uuid.uuid4().hex+'_'+label+'.png');image.save(p);return str(p)
def add_jobs(chars,pose,kind,pose_image,pose_description,count,seed,shots,bgmode,bgimage,foreground,extra,weight,bheight,cheight,with_ids=False):
 if not chars:raise gr.Error('캐릭터를 한 명 이상 선택하세요.')
 if kind!='preset' and pose_image is None:raise gr.Error('포즈 이미지를 올리세요.')
 if bgmode=='upload' and bgimage is None:raise gr.Error('합성할 배경 이미지를 올리세요.')
 try:
  e.api('/sdapi/v1/options',timeout=5)
  available={x['name'] for x in e.api('/sdapi/v1/loras',timeout=10)}
  missing=[e.PRESETS['characters'][a]['name'] for a in chars if e.PRESETS['characters'][a]['lora'] not in available]
  if missing:raise ValueError('Forge에 등록되지 않은 모델: '+', '.join(missing))
 except Exception as exc:raise gr.Error(str(exc))
 posepath=copy_upload(pose_image,'pose') if kind!='preset' else None;bgpath=copy_upload(bgimage,'background') if bgmode=='upload' else None
 ids=[]
 for char in chars:
  for i in range(int(count)):
   value=e.PRESETS['characters'][char]['recommended_seed']+i if int(seed)==-2 else (int(seed)+i if int(seed)>=0 else -1)
   p={'character':char,'pose':pose,'pose_kind':kind,'pose_path':posepath,'pose_description':pose_description,'seed':value,'shots':shots,'background_mode':bgmode,'background_path':bgpath,'foreground':foreground,'extra':extra,'weight':float(weight),'b_height':float(bheight),'c_height':float(cheight),'lo':35,'hi':210,'preset_version':e.PRESETS['version']}
   ids.append(e.enqueue(p))
 result=(f'{len(ids)}개 작업을 등록했습니다. 아래 작업 현황에서 결과를 확인하세요.',*refresh(ids[0]))
 return (*result,ids) if with_ids else result
def load_job(id):
 j=e.job(id) if id else None
 if not j:return [],'선택한 작업이 없습니다.',None,None,None
 p=json.loads(j['params']);result=json.loads(j['result']) if j['result'] else None
 rawpath=e.RESULTS/id/'generated_raw.png'
 if not rawpath.exists():rawpath=e.RESULTS/id/'raw.png'
 if result:rawpath=Path(result['raw'])
 if not rawpath.exists():return [],j['error'] or j['stage'],None,None,None
 raw=Image.open(rawpath).convert('RGB')
 if result:
  fm=Image.open(result['figure_mask']).convert('L');om=Image.open(result['omission_mask']).convert('L')
 else:
  try:fm=e.isolate(raw).getchannel('A')
  except ValueError:fm=Image.new('L',raw.size,0)
  om=Image.new('L',raw.size,0)
 gallery=[(str(rawpath),'생성 원본')]
 if result:
  for s in ['B','C']:
   if result.get(s):gallery.append((result[s],s+'컷'))
  for s in ['B_cutout','C_cutout']:
   if result.get(s):gallery.append((result[s],s.replace('_cutout','')+' 투명 인물'))
 warnings='\n'.join(result.get('warnings',[])) if result else ''
 description=f"현재 불러온 결과: {id}\n{e.PRESETS['characters'][p['character']]['name']} / 시드 {p['seed']} / {STATUS.get(j['status'],j['status'])}\n{warnings}\n{j['error'] or ''}".strip()
 return gallery,description,editor_value(raw,fm),editor_value(raw,om,(239,115,56)),id
def save_review(id,figure_editor,omit_editor):
 j=e.job(id) if id else None
 if not j:raise gr.Error('검수할 생성 결과를 먼저 불러오세요.')
 if j['status']=='running':raise gr.Error('생성이 끝난 뒤 수정하세요.')
 result=json.loads(j['result']) if j['result'] else {};rawpath=Path(result.get('raw',e.RESULTS/id/'generated_raw.png'))
 if not rawpath.is_file():raise gr.Error('저장된 생성 원본이 없습니다. 먼저 생성하세요.')
 raw=Image.open(rawpath).convert('RGB');p=json.loads(j['params'])
 try:
  fm=editor_mask(figure_editor,raw.size);om=editor_mask(omit_editor,raw.size)
  if fm.getbbox() is None:raise ValueError('인물 영역이 비어 있습니다. 인물 영역을 파란색으로 칠하세요.')
  folder=e.RESULTS/id/'revisions'/uuid.uuid4().hex[:8];new=e.compose_files(raw,p,folder,fm,om);new['review_revision']=True;e.update(id,result=json.dumps(new,ensure_ascii=False),status='review',stage='마스크 수정·재합성 완료',error='')
 except Exception as exc:raise gr.Error(str(exc))
 return load_job(id)
def decision(id,value):
 if not id or not e.job(id):raise gr.Error('작업을 선택하세요.')
 if e.job(id)['status'] not in ['review','approved','rejected']:raise gr.Error('완료된 결과만 검수할 수 있습니다.')
 e.update(id,status=value,stage='검수 승인' if value=='approved' else '재작업 표시');gr.Info('현재 불러온 결과의 검수 상태를 저장했습니다.');return '검수 상태를 저장했습니다.',*refresh(id)
def cancel(id):
 j=e.job(id) if id else None
 if not j:raise gr.Error('작업을 선택하세요.')
 if j['status']=='queued':e.update(id,status='cancelled',cancel=1,stage='대기 작업 취소')
 elif j['status']=='running':e.update(id,cancel=1,stage='현재 생성 후 취소 예정')
 else:raise gr.Error('대기 또는 실행 중인 작업만 취소할 수 있습니다.')
 return '취소 요청을 저장했습니다. 실행 중인 GPU 요청은 끝난 뒤 후속 단계를 중지합니다.',*refresh(id)
def retry(id):
 j=e.job(id) if id else None
 if not j:raise gr.Error('작업을 선택하세요.')
 new=e.enqueue(json.loads(j['params']));return '같은 설정으로 재시도 작업을 등록했습니다.',*refresh(new)
def export(id,include_records=True):
 j=e.job(id) if id else None
 if not j or not j['result']:raise gr.Error('저장할 결과를 먼저 선택하세요.')
 params=json.loads(j['params'])
 if params.get('foreground',False) or (params.get('background_mode','guide')=='guide' and not params.get('background_path')):
  remove_comparison(id);j=e.job(id)
 result=json.loads(j['result']);folder=e.RESULTS/id;p=folder/('전체기록.zip' if include_records else 'B_C_이미지.zip')
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  keys=['raw','B','C','B_cutout','C_cutout','figure_mask','omission_mask'] if include_records else ['B','C']
  for k in keys:
   if result.get(k) and Path(result[k]).is_file():z.write(result[k],k+'.png')
  if include_records:
   for fn in ['request.json','generation.json','pose.png','pose_input.png','generated_raw.png']:
    if (folder/fn).exists():z.write(folder/fn,fn)
   params=json.loads(j['params'])
   for key in ['background_path','pose_path']:
    if params.get(key) and Path(params[key]).is_file():z.write(params[key],key+'.png')
   z.writestr('preset.json',json.dumps(e.PRESETS,ensure_ascii=False,indent=2))
   z.write(e.ROOT/'models'/'manifest.json','models_manifest.json')
   z.writestr('job.json',json.dumps({'id':id,'status':j['status'],'params':json.loads(j['params']),'result':result},ensure_ascii=False,indent=2))
 return str(p)
def configure(url,forge,models):
 try:e.save_settings(url,forge,models)
 except Exception as exc:raise gr.Error(str(exc))
 return '연결 설정을 저장했습니다.',e.preflight()
def start_forge():
 try:return forge_control.start()
 except Exception as exc:raise gr.Error(str(exc))
def load_example():
 folder=e.ROOT/'examples';id='example_preview';params={'character':'char_a','seed':29200,'shots':'both','foreground':True,'background_mode':'guide','b_height':125,'c_height':33.6,'lo':35,'hi':210}
 target=e.RESULTS/id;target.mkdir(exist_ok=True)
 for fn in ['raw.png','figure_mask.png','omission_mask.png','B.png','C.png']:shutil.copy2(folder/fn,target/fn)
 result={k:str(target/v) for k,v in [('raw','raw.png'),('figure_mask','figure_mask.png'),('omission_mask','omission_mask.png'),('B','B.png'),('C','C.png')]}
 with e.LOCK,e.connect_db() as c:c.execute('INSERT OR REPLACE INTO jobs (id,created,status,params,result,stage) VALUES (?,?,?,?,?,?)',(id,time.strftime('%Y-%m-%d %H:%M:%S'),'review',json.dumps(params),json.dumps(result),'기존 생성 예시 · 새 생성 아님'))
 return refresh(id)

def load_view(id):
 gallery,description,figure,omit,current=load_job(id)
 images={label:path for path,label in gallery}
 return images.get('생성 원본'),images.get('B컷'),images.get('C컷'),description,figure,omit,current,None

def save_view(id,figure,omit):
 save_review(id,figure,omit)
 return load_view(id)

def export_view(id,records):
 path=export(id,records)
 view=list(load_view(id));view[-1]=path
 return tuple(view)

def remove_comparison(id):
 j=e.job(id) if id else None
 if not j or not j['result']:raise gr.Error('결과를 먼저 불러오세요.')
 if j['status'] not in ['review','approved','rejected']:raise gr.Error('생성이 끝난 결과에서 제거할 수 있습니다.')
 params=json.loads(j['params']);result=json.loads(j['result'])
 guide=params.get('background_mode','guide')=='guide' and not params.get('background_path')
 if not params.get('foreground',False) and not guide:
  gr.Info('현재 결과에는 비교 인물과 원근 격자가 없습니다.');return load_view(id)
 params['foreground']=False
 if guide:params['background_mode']='white'
 raw=Image.open(result['raw']).convert('RGB')
 fm=Image.open(result['figure_mask']).convert('L');om=Image.open(result['omission_mask']).convert('L')
 folder=e.RESULTS/id/'revisions'/('clean_'+uuid.uuid4().hex[:8])
 new=e.compose_files(raw,params,folder,fm,om)
 new.update(review_revision=True,comparison_removed=True,guide_removed=guide or result.get('guide_removed',False))
 e.dump(folder/'result.json',new)
 e.update(id,params=json.dumps(params,ensure_ascii=False),result=json.dumps(new,ensure_ascii=False),status='review',stage='구도 가이드 제거 · 검수 대기',error='')
 gr.Info('B/C컷에서 비교 인물과 원근 격자를 제거했습니다. 원본 인물과 마스크는 유지됩니다.')
 return load_view(id)

def submit_view(chars,pose,kind,pose_image,pose_description,count,mode,number,shots,bgmode,bgimage,foreground,extra,weight,bheight,cheight):
 try:seed=seed_value(mode,number)
 except ValueError as exc:raise gr.Error(str(exc))
 message,table,selected,ids=add_jobs(chars,pose,kind,pose_image,pose_description,count,seed,shots,bgmode,bgimage,foreground,extra,weight,bheight,cheight,with_ids=True)
 return message,table,selected,{'ids':ids,'opened':False},progress_text(ids),None

def poll_view(selected,batch,automatic,current):
 table,choice=refresh(selected);ids=(batch or {}).get('ids',[])
 progress=progress_text(ids)
 untouched=[gr.skip() for _ in range(9)]
 if ids and automatic and not batch.get('opened'):
  rows=[e.job(i) for i in ids];rows=[r for r in rows if r]
  finished=bool(rows) and all(r['status'] not in ['queued','running'] for r in rows)
  ready=next((r for r in rows if r['result']),None)
  if finished and ready:
   batch={**batch,'opened':True};choice=refresh(ready['id'])[1]
   return table,choice,progress,batch,*load_view(ready['id']),gr.update(selected='review')
 return table,choice,progress,batch,*untouched

def review_decision(id,value):
 decision(id,value)
 return load_view(id)

def selected_changed(selected,current):
 different=bool(current and selected!=current)
 if different:
  return '다른 작업을 선택했습니다. **선택한 결과 불러오기**를 눌러야 검수·저장 대상이 바뀝니다.'
 return '검수·저장은 아래 **현재 불러온 결과**에 적용됩니다.'

def build():
 cfg=e.settings()
 with gr.Blocks(title='웹툰 캐릭터 스튜디오 1.3') as ui:
  gr.Markdown('# 웹툰 캐릭터 스튜디오\n캐릭터를 만들고, 컷 안의 크기와 생략할 선을 확인하세요. **1.3**')
  with gr.Tabs() as tabs:
   with gr.Tab('1 · 캐릭터 생성',id='generate'):
    with gr.Row():
     with gr.Column():
      chars=gr.CheckboxGroup(CHARS,value=['char_a'],label='캐릭터 · 여러 명 선택 가능')
      pose=gr.Dropdown(POSES,value='recommended',label='포즈')
      with gr.Accordion('내 포즈 이미지 사용',open=False):
       kind=gr.Radio([('추천·선택 포즈','preset'),('3D·일반 이미지','image'),('이미 추출한 OpenPose','openpose')],value='preset',label='포즈 입력')
       pose_img=gr.Image(type='pil',sources=['upload'],label='포즈 이미지',height=180)
       pose_desc=gr.Textbox(label='자세 설명',placeholder='예: sitting, crossed arms, holding sword')
      shots=gr.Radio([('B·C 둘 다','both'),('B컷','B'),('C컷','C')],value='both',label='생성할 컷')
      count=gr.Slider(1,4,value=1,step=1,label='캐릭터별 후보 수')
      mode=gr.Radio([('기존 검증 설정','verified'),('새로운 결과','random'),('직접 지정','manual')],value='verified',label='결과 선택 방식')
      seed=gr.Number(value=29200,precision=0,label='시드 · 같은 조건 비교용 숫자',visible=False)
      gr.Markdown('기존 검증 설정은 캐릭터별로 저장한 시드를 사용합니다. 새로운 결과는 매번 다른 시드를 기록합니다.')
      with gr.Accordion('묘사와 캐릭터 강도',open=False):
       extra=gr.Textbox(label='추가 묘사');weight=gr.Slider(.5,1.2,value=.9,step=.05,label='캐릭터 강도 · 권장 0.90')
     with gr.Column():
      gr.Markdown('### 컷 안의 인물 크기\n슬라이더를 움직여 B/C 비중을 비교하세요. 아래 도형은 크기 설명용입니다.')
      bh=gr.Slider(60,180,value=125,step=1,label='B 인물 높이 / 컷 높이 (%)')
      ch=gr.Slider(15,55,value=33.6,step=.1,label='C 인물 높이 / 컷 높이 (%)')
      with gr.Row():
       preview_b=gr.Image(label='B 크기 미리보기',interactive=False,height=270)
       preview_c=gr.Image(label='C 크기 미리보기',interactive=False,height=270)
      preview_note=gr.Markdown()
      bgmode=gr.Dropdown([('흰 배경 · 기본','white'),('원근 격자 · 저장 시 제거','guide'),('배경 업로드','upload')],value='white',label='합성 배경')
      bgimg=gr.Image(type='pil',sources=['upload'],label='합성 배경 이미지',height=150,visible=False)
      foreground=gr.Checkbox(value=False,label='구도 확인용 비교 인물 표시 · 저장 시 자동 제거')
    add=gr.Button('선택한 조건으로 생성',variant='primary')
    message=gr.Textbox(label='작업 안내',interactive=False)
    automatic=gr.Checkbox(value=True,label='이번 생성이 끝나면 결과 검수 화면 열기')
    progress=gr.Markdown()
    gr.Markdown('캐릭터별 개별 LoRA · 기본 강도 0.90 · 별도 스타일 LoRA 미사용 · 캐릭터I 모델 미확보')
    table=gr.Dataframe(headers=['작업 ID','캐릭터','상태','진행','시드','메시지'],datatype=['str']*6,interactive=False,label='작업 현황',wrap=True)
    batch=gr.State({'ids':[],'opened':False})
   with gr.Tab('2 · 결과 검수',id='review'):
    selected=gr.Dropdown([],label='작업 선택');load=gr.Button('선택한 결과 불러오기',variant='primary')
    selection_note=gr.Markdown('검수할 결과를 먼저 불러오세요.')
    details=gr.Textbox(label='현재 불러온 결과 · 검수와 저장 대상',interactive=False,lines=3)
    gr.Markdown('**B와 C는 같은 생성 원본을 사용합니다.** B는 크게 배치하고, C는 작게 배치하며 얼굴 내부선 생략·잔선 정리를 적용합니다.')
    with gr.Row():
     rawview=gr.Image(label='생성 원본',interactive=False,height=500)
     bview=gr.Image(label='B컷 · 크게 보기',interactive=False,height=500)
     cview=gr.Image(label='C컷 · 거리감과 실루엣',interactive=False,height=500)
    current=gr.State(None)
    remove=gr.Button('구도 가이드 제거 · 비교 인물·격자 없이 재합성')
    gr.Markdown('비교용 앞쪽 남성과 원근 격자를 제거합니다. ZIP 저장 시에도 자동 제거됩니다. 업로드한 배경은 유지됩니다. 원본과 현재 마스크로 다시 합성합니다. 편집 중인 영역은 먼저 다시 합성해 저장하세요.')
    with gr.Accordion('영역 수정 · GPU 재생성 없이 다시 합성',open=False):
     gr.Markdown('**파랑:** 인물로 남길 영역. **주황:** C컷에서 눈·코·입 내부선을 생략할 영역. 원본이 비쳐 보이므로 선을 확인하며 칠하세요. 지우개로 영역을 줄일 수 있습니다. 머리·귀·턱 외곽은 남겨주세요.')
     with gr.Row():
      figure=gr.ImageEditor(type='pil',image_mode='RGBA',label='인물 영역 · 파랑',sources=[],transforms=[],height=540,brush=gr.Brush(colors=[('#209ce8',100/255)],color_mode='fixed'),format='png')
      omit=gr.ImageEditor(type='pil',image_mode='RGBA',label='C컷 생략 영역 · 주황',sources=[],transforms=[],height=540,brush=gr.Brush(colors=[('#ef7338',100/255)],color_mode='fixed'),format='png')
     save=gr.Button('수정한 영역으로 다시 합성')
    with gr.Row():
     approve=gr.Button('현재 결과 검수 승인');reject=gr.Button('현재 결과 재작업 표시');again=gr.Button('선택 작업 같은 설정으로 재시도');stop=gr.Button('선택 작업 취소')
    gr.Markdown('### 현재 결과 저장\n검수 승인은 사용자가 확인했다는 표시입니다. 저장 파일에는 아래에서 선택한 내용만 담습니다.')
    records=gr.Checkbox(value=False,label='원본·투명 인물·마스크·생성 기록까지 포함')
    pack=gr.Button('현재 결과 ZIP 저장',variant='primary');download=gr.File(label='저장할 결과 파일',interactive=False)
   with gr.Tab('3 · 환경 점검',id='environment'):
    gr.Markdown('Forge 설치 폴더와 제공된 models 폴더를 연결하세요. 필요한 파일과 모델 등록 상태를 확인합니다.')
    url=gr.Textbox(value=cfg['api_url'],label='Forge 연결 주소');forge=gr.Textbox(value=cfg['forge_dir'],label='Forge 설치 폴더 · launch.py가 있는 곳');models=gr.Textbox(value=cfg['models_dir'],label='모델 폴더')
    config=gr.Button('설정 저장 및 점검');launch=gr.Button('지정한 Forge 실행');check=gr.Button('다시 점검');envmsg=gr.Textbox(label='환경 안내',interactive=False);envtable=gr.Dataframe(headers=['항목','상태','내용'],interactive=False,label='필수 파일·연결 상태',wrap=True)
   with gr.Tab('4 · 사용 안내',id='help'):
    gr.Markdown('''### 사용 순서
1. 환경 점검에서 Forge와 모델을 연결합니다.
2. 캐릭터·포즈를 고르고 크기 미리보기에서 B/C 비중을 확인합니다.
3. 생성하면 작업 단계와 완료 수가 표시됩니다. 자동 열기를 켜면 이번 작업이 모두 끝난 뒤 결과 화면이 열립니다.
4. 원본·B·C를 비교하고 손·무기·원작 특징·잘림을 확인합니다. 필요하면 영역을 수정해 다시 합성합니다.
5. 현재 결과를 승인하거나 재작업 표시하고 ZIP으로 저장합니다. 기본 저장은 B/C 이미지입니다.

### 확인할 점
- C컷은 얼굴보다 머리·체격·귀·의상·장비의 실루엣이 읽히는지 봅니다.
- 크기 미리보기는 설명용 도형이며 실제 캐릭터 예측이 아닙니다. 넓은 자세는 프레임에 맞춰 더 작아질 수 있습니다.
- 기본 원근 가이드와 전경 인물은 구도 확인용입니다.
- 영역 편집의 색은 화면 표시용입니다. 저장되는 마스크는 남김/생략을 구분하는 이진 마스크입니다.
- 생성 도중 취소는 현재 GPU 생성이 끝난 뒤 적용됩니다. 창을 닫아 중단된 작업은 다음 실행에서 실패로 표시되며 재시도할 수 있습니다.
- 브라우저만 닫으면 서버는 계속 실행됩니다. 실행 창에서 Ctrl+C로 종료합니다.
- 같은 시드도 GPU·Forge·라이브러리가 바뀌면 결과가 달라질 수 있습니다.
- 결과와 수정 이력은 results, 작업 기록과 설정은 runtime에 저장됩니다.
''')
    demo=gr.Button('기존 생성 예시 불러오기 · GPU 불필요');gr.Markdown('과거 생성물을 불러오는 기능이며 새 생성이 아닙니다.')
  review_outputs=[rawview,bview,cview,details,figure,omit,current,download]
  preview_inputs=[bh,ch,bgmode,bgimg,foreground]
  preview_outputs=[preview_b,preview_c,preview_note]
  for component in preview_inputs:component.change(layout_preview,preview_inputs,preview_outputs,show_progress='hidden')
  mode.change(lambda m:gr.update(visible=m=='manual'),mode,seed)
  bgmode.change(lambda m:gr.update(visible=m=='upload'),bgmode,bgimg)
  inputs=[chars,pose,kind,pose_img,pose_desc,count,mode,seed,shots,bgmode,bgimg,foreground,extra,weight,bh,ch]
  add.click(submit_view,inputs,[message,table,selected,batch,progress,download])
  load.click(load_view,selected,review_outputs)
  selected.change(selected_changed,[selected,current],selection_note)
  current.change(selected_changed,[selected,current],selection_note)
  save.click(save_view,[current,figure,omit],review_outputs)
  remove.click(remove_comparison,current,review_outputs)
  approve.click(lambda i:review_decision(i,'approved'),current,review_outputs)
  reject.click(lambda i:review_decision(i,'rejected'),current,review_outputs)
  stop.click(cancel,selected,[message,table,selected]).then(lambda:gr.Info('취소 요청을 저장했습니다. 실행 중인 생성은 끝난 뒤 취소됩니다.'))
  again.click(retry,selected,[message,table,selected]).then(lambda:gr.Info('같은 설정의 새 작업을 등록했습니다.'))
  records.change(lambda:None,None,download)
  pack.click(export_view,[current,records],review_outputs)
  config.click(configure,[url,forge,models],[envmsg,envtable]);launch.click(start_forge,None,envmsg);check.click(e.preflight,None,envtable)
  demo.click(load_example,None,[table,selected]).then(load_view,selected,review_outputs).then(lambda:gr.update(selected='review'),None,tabs)
  timer=gr.Timer(3)
  timer.tick(poll_view,[selected,batch,automatic,current],[table,selected,progress,batch,*review_outputs,tabs],show_progress='hidden')
  ui.load(refresh,None,[table,selected]);ui.load(layout_preview,preview_inputs,preview_outputs)
  ui.load(progress_text,None,progress)
 return ui

if __name__=='__main__':
 e.start_worker();app=build();port=int(os.environ.get('STUDIO_UI_PORT','7870'))
 app.queue(default_concurrency_limit=4).launch(server_name='127.0.0.1',server_port=port,inbrowser='--no-browser' not in sys.argv,allowed_paths=[str(e.ROOT)],show_error=True,theme=gr.themes.Soft(primary_hue='indigo'),css='.gradio-container{max-width:1500px!important} footer{display:none!important}')
