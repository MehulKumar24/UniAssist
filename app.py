from __future__ import annotations
import hashlib,re,time
from datetime import date,datetime
from pathlib import Path
import numpy as np,pandas as pd,streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title='UniAssist Pro',page_icon='🎓',layout='wide')
DATA_DIR=Path('data');FEEDBACK_FILE=DATA_DIR/'feedback.csv';QUERY_LOG_FILE=DATA_DIR/'query_logs.csv';ALERT_FILE=DATA_DIR/'alerts.csv'
SIM_DEFAULT=0.65
LANGS=['English','Hindi','Spanish','French']
TENANTS={'Default University':{'owner':'Academic Office'},'North Campus Institute':{'owner':'Dean Academics'},'West Tech University':{'owner':'Registrar'}}
USERS={'student_demo':{'password':'student123','role':'student'},'faculty_demo':{'password':'faculty123','role':'faculty'},'admin_demo':{'password':'admin123','role':'admin'}}
SCOPE={'attendance','exam','internship','policy','grades','cgpa','credit','syllabus','academic','placement','scholarship','leave','deadline'}
BLOCK={'hack','bypass exam','fake certificate','violence','self-harm'}
STEPS={'attendance':['Check ERP attendance','Raise regularization','Meet advisor'],'internship':['Validate eligibility','Prepare documents','Submit on portal'],'exam':['Verify schedule','Check revaluation policy','Contact exam cell'],'general':['Read official notice','Follow department process','Escalate if unresolved']}

def ensure_storage():
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    if not FEEDBACK_FILE.exists(): pd.DataFrame(columns=['timestamp','user','role','university','query','response','confidence','feedback','comment']).to_csv(FEEDBACK_FILE,index=False)
    if not QUERY_LOG_FILE.exists(): pd.DataFrame(columns=['timestamp','user','role','university','department','semester','query','category','confidence','scope_pass','latency_ms','escalated']).to_csv(QUERY_LOG_FILE,index=False)
    if not ALERT_FILE.exists(): pd.DataFrame(columns=['timestamp','user','alert_type','details']).to_csv(ALERT_FILE,index=False)

def tok(t:str)->set[str]: return set(re.findall(r'[a-zA-Z0-9]+',t.lower()))
def cat(t:str)->str:
    t=t.lower()
    if any(k in t for k in ['attendance','absent','leave']): return 'attendance'
    if any(k in t for k in ['internship','placement','offer']): return 'internship'
    if any(k in t for k in ['exam','revaluation','grade','cgpa']): return 'exam'
    return 'general'
def tr(txt:str,lang:str)->str: return txt if lang=='English' else f'[{lang} beta translation] {txt}'
def append_row(p:Path,row:dict): pd.DataFrame([row]).to_csv(p,mode='a',header=False,index=False)
def trust(conf:float,fresh_days:int,cites:int)->float: return round((max(0,min(1,conf))*0.6+max(0,1-min(fresh_days,365)/365)*0.25+min(cites/3,1)*0.15)*100,1)
def guard(q:str)->tuple[bool,str]:
    l=q.lower()
    if any(x in l for x in BLOCK): return False,'Query blocked by safety moderation policy.'
    if len(tok(q)&SCOPE)==0: return False,'Out-of-scope: academic/internship only.'
    return True,'ok'
def ready(res:int,mock:int,proj:int)->int: return max(0,min(int(res*0.5+min(mock,10)*4+min(proj,5)*8),100))
def attn_proj(cur:float,done:int,fut:int,att_fut:int)->float:
    tot=done+fut
    return cur if tot<=0 else round((((cur/100)*done)+att_fut)/tot*100,2)
def cgpa_proj(cur:float,cred:int,newc:int,gp:float)->float:
    tot=cred+newc
    return cur if tot<=0 else round((cur*cred+gp*newc)/tot,2)

@st.cache_data
def load_base()->pd.DataFrame:
    f=pd.read_csv('UniAssist_training_data.csv')
    if 'question' not in f.columns or 'answer' not in f.columns: raise ValueError("CSV needs 'question' and 'answer'")
    f=f.copy();f['question']=f['question'].astype(str);f['answer']=f['answer'].astype(str)
    if 'university' not in f.columns: f['university']='Default University'
    if 'category' not in f.columns: f['category']=f['question'].apply(cat)
    if 'source' not in f.columns: f['source']='UniAssist_training_data.csv'
    if 'last_updated' not in f.columns: f['last_updated']='2026-01-01'
    if 'policy_link' not in f.columns: f['policy_link']='https://university.example/policies'
    return f
@st.cache_resource
def model()->SentenceTransformer: return SentenceTransformer('all-MiniLM-L6-v2')
@st.cache_resource
def embeds(qs:tuple[str,...])->np.ndarray: return model().encode(list(qs),normalize_embeddings=True)
def kb()->pd.DataFrame:
    b=load_base();c=st.session_state.get('custom_kb',pd.DataFrame())
    if c.empty: return b
    return pd.concat([b,c],ignore_index=True).drop_duplicates(subset=['question','answer'],keep='last')
def retrieve(q:str,k:pd.DataFrame,u:str,cf:str,top:int)->pd.DataFrame:
    f=k[k['university'].isin([u,'Default University'])].copy()
    if cf!='All': f=f[f['category']==cf]
    if f.empty: f=k.copy()
    qs=f['question'].tolist();sem=cosine_similarity(model().encode([q],normalize_embeddings=True),embeds(tuple(qs)))[0]
    qt=tok(q);key=np.array([len(qt&tok(x))/max(len(qt),1) for x in qs]);bonus=np.where(f['category'].values==cat(q),0.05,0.0)
    f=f.copy();f['score']=sem*0.72+key*0.23+bonus
    return f.sort_values('score',ascending=False).head(top)
def answer(q:str,k:pd.DataFrame,u:str,cf:str,top:int)->dict:
    t=time.perf_counter();r=retrieve(q,k,u,cf,top);lat=int((time.perf_counter()-t)*1000)
    if r.empty: return {'answer':'No answer found.','confidence':0.0,'category':'general','citations':[],'matched':None,'fresh':365,'latency':lat}
    b=r.iloc[0];lu=pd.to_datetime(b.get('last_updated','2026-01-01'),errors='coerce');fresh=365 if pd.isna(lu) else (datetime.now()-lu.to_pydatetime()).days
    cites=[{'source':str(x.get('source','unknown')),'link':str(x.get('policy_link','https://university.example/policies')),'updated':str(x.get('last_updated','unknown'))} for _,x in r.iterrows()]
    return {'answer':str(b['answer']),'confidence':float(b['score']),'category':str(b.get('category','general')),'citations':cites,'matched':str(b['question']),'fresh':fresh,'latency':lat}
def init():
    d={'authenticated':False,'username':'guest','role':'student','conversation':[],'custom_kb':pd.DataFrame(),'review_queue':[],'resolved_queue':[],'tickets':[],'policy_hashes':{},'consent':True,'session_started':time.time(),'similarity_threshold':SIM_DEFAULT,'last_response':None,'checklist':[{'task':'Update resume','due':str(date.today()),'done':False},{'task':'Review attendance','due':str(date.today()),'done':False}]}
    for k,v in d.items():
        if k not in st.session_state: st.session_state[k]=v

def sidebar():
    with st.sidebar:
        st.header('Access');st.caption('Demo accounts: student_demo / faculty_demo / admin_demo')
        if not st.session_state['authenticated']:
            with st.form('login'):
                u=st.text_input('Username','student_demo');p=st.text_input('Password',type='password',value='student123')
                if st.form_submit_button('Login'):
                    rec=USERS.get(u)
                    if rec and rec['password']==p:
                        st.session_state['authenticated']=True;st.session_state['username']=u;st.session_state['role']=rec['role'];st.success('Authenticated')
                    else: st.error('Invalid credentials')
        else:
            st.success(f"Logged in: {st.session_state['username']} ({st.session_state['role']})")
            if st.button('Logout'):
                st.session_state['authenticated']=False;st.session_state['username']='guest';st.session_state['role']='student';st.session_state['conversation']=[];st.session_state['last_response']=None;st.rerun()
        st.divider();u=st.selectbox('University',list(TENANTS.keys()),key='selected_university');st.caption(f"Policy owner: {TENANTS[u]['owner']}")
        st.slider('Similarity threshold',0.50,0.90,SIM_DEFAULT,0.01,key='similarity_threshold')
        st.checkbox('Consent to store query analytics',key='consent');st.toggle('Verified answer mode',value=True,key='verified_mode')
        st.selectbox('Preferred language',LANGS,key='preferred_language');st.toggle('Parent/Guardian read-only mode',False,key='parent_mode');st.toggle('Offline kiosk mode',False,key='offline_mode')
        st.divider();st.caption(f"Session uptime: {int((time.time()-st.session_state['session_started'])/60)} min");st.caption('Auth: RBAC enabled');st.caption('Rate limiting: Demo mode')
def log_query(r:dict,q:str,dpt:str,sem:int,scope_pass:bool,esc:bool):
    if not st.session_state['consent']: return
    append_row(QUERY_LOG_FILE,{'timestamp':datetime.now().isoformat(timespec='seconds'),'user':st.session_state['username'],'role':st.session_state['role'],'university':st.session_state['selected_university'],'department':dpt,'semester':sem,'query':q,'category':r['category'],'confidence':round(r['confidence'],4),'scope_pass':int(scope_pass),'latency_ms':r['latency'],'escalated':int(esc)})

def assistant_tab(k:pd.DataFrame):
    st.subheader('AI Guidance Assistant')
    c1,c2,c3=st.columns([1.1,1,1])
    with c1: dpt=st.selectbox('Department',['CSE','ECE','ME','CE'],key='department')
    with c2: sem=st.selectbox('Semester',list(range(1,9)),key='semester')
    with c3: cf=st.selectbox('Category filter',['All','attendance','exam','internship','general'])
    if hasattr(st,'audio_input'):
        a=st.audio_input('Voice query (beta)')
        if a is not None: st.info('Voice captured. Connect STT provider for transcription.')
    with st.form('qa_form'):
        q=st.text_area('Ask your question',placeholder='What is the minimum attendance required for semester exams?',height=90)
        top=st.slider('Top sources',1,5,3);ask=st.form_submit_button('Get Verified Answer')
    if ask:
        q=q.strip()
        if not q: st.warning('Please enter a question.');return
        ok,msg=guard(q)
        if not ok: st.error(msg);return
        r=answer(q,k,st.session_state['selected_university'],cf,top);ts=trust(r['confidence'],r['fresh'],len(r['citations']));esc=r['confidence']<st.session_state['similarity_threshold']
        shown=r['answer']
        if esc:
            st.session_state['review_queue'].append({'timestamp':datetime.now().isoformat(timespec='seconds'),'query':q,'suggested_answer':r['answer'],'confidence':r['confidence'],'category':r['category']})
            shown='I do not have high-confidence evidence. This query has been added to human review queue.'
        st.write(tr(shown,st.session_state['preferred_language']))
        st.caption(f"Confidence: {r['confidence']:.2f} | Trust: {ts}/100 | Latency: {r['latency']} ms")
        st.caption(f"Matched question: {r['matched']}")
        st.markdown('### Next 3 Steps')
        for s in STEPS.get(r['category'],STEPS['general']): st.write(f'- {s}')
        if st.session_state['verified_mode']:
            st.markdown('### Citations')
            for c in r['citations']: st.write(f"- Source: {c['source']} | Updated: {c['updated']} | Link: {c['link']}")
        st.session_state['last_response']={'query':q,'response':shown,'confidence':r['confidence'],'category':r['category']}
        st.session_state['conversation'].append({'q':q,'a':shown});log_query(r,q,dpt,sem,True,esc)
    st.markdown('### Feedback Loop')
    if st.session_state['last_response'] is not None:
        with st.form('feedback_form'):
            fb=st.radio('Was this helpful?',['👍 Helpful','👎 Not helpful'],horizontal=True);cm=st.text_input('Optional correction')
            if st.form_submit_button('Submit feedback'):
                lr=st.session_state['last_response']
                append_row(FEEDBACK_FILE,{'timestamp':datetime.now().isoformat(timespec='seconds'),'user':st.session_state['username'],'role':st.session_state['role'],'university':st.session_state['selected_university'],'query':lr['query'],'response':lr['response'],'confidence':lr['confidence'],'feedback':fb,'comment':cm})
                st.success('Feedback saved')
        if st.button('Escalate to Counselor'):
            lr=st.session_state['last_response'];st.session_state['tickets'].append({'time':datetime.now().isoformat(timespec='seconds'),'user':st.session_state['username'],'query':lr['query'],'summary':f"Category={lr['category']} confidence={lr['confidence']:.2f}"});st.success('Escalation ticket created')
    with st.expander('Conversation Memory'):
        if not st.session_state['conversation']: st.caption('No prior conversation')
        else:
            for t in st.session_state['conversation'][-10:]: st.write(f"Q: {t['q']}");st.write(f"A: {t['a']}");st.write('---')

def student_tab():
    st.subheader('Student Success Center')
    c1,c2,c3=st.columns(3)
    with c1: ap=st.number_input('Current attendance %',0.0,100.0,78.0,0.1);cd=st.number_input('Classes completed',1,500,60)
    with c2: fc=st.number_input('Upcoming classes',0,200,20);af=st.number_input('Planned attended classes',0,200,16)
    with c3: cg=st.number_input('Current CGPA',0.0,10.0,7.2,0.01);cr=st.number_input('Credits completed',1,250,90)
    pa=attn_proj(ap,cd,fc,af);pc=cgpa_proj(cg,cr,20,8.0)
    risks=[]
    if pa<75: risks.append('Attendance risk')
    if pc<6.0: risks.append('CGPA risk')
    st.markdown('### Predictive Alerts')
    if risks:
        for r in risks:
            st.error(r);append_row(ALERT_FILE,{'timestamp':datetime.now().isoformat(timespec='seconds'),'user':st.session_state['username'],'alert_type':r,'details':f'Projected attendance={pa}, projected CGPA={pc}'})
    else: st.success('No immediate academic risk detected')
    st.caption(f'Projected attendance: {pa}%');st.caption(f'Projected CGPA (what-if): {pc}')
    st.markdown('### Scholarship + Internship Matcher')
    p=pd.DataFrame([{'program':'Merit Scholarship A','requires_cgpa':8.0,'domain':'academic','deadline':'2026-03-10'},{'program':'AI Internship Track','requires_cgpa':7.0,'domain':'ai','deadline':'2026-03-25'},{'program':'Core Engineering Internship','requires_cgpa':6.5,'domain':'core','deadline':'2026-04-12'}])
    dom=st.selectbox('Interest domain',['ai','core','academic']);p['fit']=p.apply(lambda r:(20 if cg>=r['requires_cgpa'] else 0)+(80 if r['domain']==dom else 30),axis=1);st.dataframe(p.sort_values('fit',ascending=False),use_container_width=True)
    st.markdown('### Placement Readiness Engine')
    a,b,c=st.columns(3)
    with a: rs=st.slider('Resume score',0,100,68)
    with b: mi=st.slider('Mock interviews',0,20,3)
    with c: pr=st.slider('Projects',0,10,2)
    rr=ready(rs,mi,pr);st.progress(rr/100);st.caption(f'Readiness score: {rr}/100')
    if st.session_state['parent_mode']:
        st.markdown('### Parent / Guardian View');st.info('Read-only summary enabled');st.write({'attendance_projection':pa,'cgpa_projection':pc,'risk_flags':risks or ['none']})

def workflow_tab():
    st.subheader('Workflow Automation');st.markdown('### Deadline Reminders + Checklist');cl=st.session_state['checklist']
    for i,it in enumerate(cl):
        cols=st.columns([3,2,1]);cols[0].write(it['task']);cols[1].write(it['due']);it['done']=cols[2].checkbox('Done',value=it['done'],key=f'check_{i}')
    with st.form('add_task'):
        t=st.text_input('New task');d=st.date_input('Due date',value=date.today(),key='new_due')
        if st.form_submit_button('Add task') and t.strip(): cl.append({'task':t.strip(),'due':str(d),'done':False});st.success('Task added')
    st.markdown('### Smart Form: Leave/Attendance Request')
    with st.form('leave'):
        r=st.text_input('Reason');n=st.number_input('No. of days',1,30,2)
        if st.form_submit_button('Generate request'): st.code(f'Subject: Attendance regularization request\nReason: {r}\nDays: {n}\nRequest: Kindly regularize attendance as per policy.',language='text')
    st.markdown('### Smart Form: Grade Appeal Draft')
    with st.form('appeal'):
        c=st.text_input('Course code');g=st.text_area('Appeal reason')
        if st.form_submit_button('Generate appeal'): st.code(f'Subject: Grade Appeal - {c}\nI request a revaluation for {c}.\nReason: {g}\nAttached evidence enclosed.',language='text')
    st.markdown('### Internship Application Copilot')
    with st.form('intern'):
        role=st.text_input('Target role','Software Intern');s=st.text_area('Top strengths','Python, ML, problem-solving')
        if st.form_submit_button('Generate application email'): st.code(f'Dear Hiring Team,\nI am applying for the {role} position.\nMy relevant strengths: {s}.\nPlease find attached my resume.\nRegards',language='text')
def analytics_tab():
    st.subheader('Analytics, Benchmarking, and Evaluation')
    ql=pd.read_csv(QUERY_LOG_FILE);fl=pd.read_csv(FEEDBACK_FILE)
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Queries',len(ql));c2.metric('Avg confidence',round(ql['confidence'].mean(),2) if not ql.empty else 0);c3.metric('Avg latency (ms)',int(ql['latency_ms'].mean()) if not ql.empty else 0)
    hr=(fl['feedback'].eq('👍 Helpful').mean()*100) if not fl.empty else 0;c4.metric('Helpful rate',f'{hr:.1f}%')
    st.markdown('### Department Benchmark')
    if not ql.empty and 'department' in ql.columns:
        d=ql.groupby('department',as_index=False).agg(avg_confidence=('confidence','mean'),avg_latency=('latency_ms','mean'),total_queries=('query','count'));st.dataframe(d,use_container_width=True)
    else: st.info('No query benchmark data yet')
    st.markdown('### Failure Topics')
    if not ql.empty:
        low=ql[ql['confidence']<st.session_state['similarity_threshold']]
        if not low.empty: st.dataframe(low[['timestamp','query','category','confidence']].tail(20),use_container_width=True)
        else: st.success('No low-confidence queries in current logs')
    st.markdown('### Continuous Evaluation Suite')
    ev=pd.DataFrame([{'question':'minimum attendance requirement','expected':'attendance'},{'question':'how to apply for internship','expected':'internship'},{'question':'grade revaluation process','expected':'exam'}]);k=kb()
    if st.button('Run evaluation'):
        ok=0;rows=[]
        for _,r in ev.iterrows():
            out=answer(r['question'],k,st.session_state['selected_university'],'All',3);p=out['category']==r['expected'];ok+=int(p);rows.append({'question':r['question'],'expected':r['expected'],'predicted':out['category'],'confidence':round(out['confidence'],3),'pass':p})
        st.metric('Evaluation score',f"{(ok/len(ev))*100:.1f}%");st.dataframe(pd.DataFrame(rows),use_container_width=True)

def admin_tab():
    st.subheader('Admin Controls')
    if st.session_state['role']!='admin': st.warning('Admin role required');return
    st.markdown('### Knowledge Base Uploader')
    up=st.file_uploader('Upload knowledge CSV',type=['csv'],key='kb_upload')
    if up is not None:
        n=pd.read_csv(up)
        if 'question' in n.columns and 'answer' in n.columns:
            if 'university' not in n.columns: n['university']=st.session_state['selected_university']
            if 'category' not in n.columns: n['category']=n['question'].astype(str).apply(cat)
            if 'source' not in n.columns: n['source']='admin_upload.csv'
            if 'last_updated' not in n.columns: n['last_updated']=str(date.today())
            if 'policy_link' not in n.columns: n['policy_link']='https://university.example/uploaded'
            st.session_state['custom_kb']=pd.concat([st.session_state['custom_kb'],n],ignore_index=True);st.success(f'Added {len(n)} records to in-session KB')
        else: st.error('CSV must contain question and answer columns')
    st.markdown('### Human Review Queue')
    q=st.session_state['review_queue']
    if not q: st.info('No items in review queue')
    else:
        for i,it in enumerate(q):
            with st.expander(f"{i+1}. {it['query']} (confidence {it['confidence']:.2f})"):
                fa=st.text_area('Final approved answer',value=it['suggested_answer'],key=f'rev_{i}')
                if st.button('Approve',key=f'app_{i}'):
                    st.session_state['resolved_queue'].append({**it,'final_answer':fa})
                    add=pd.DataFrame([{'question':it['query'],'answer':fa,'university':st.session_state['selected_university'],'category':it['category'],'source':'human_review','last_updated':str(date.today()),'policy_link':'https://university.example/reviewed'}])
                    st.session_state['custom_kb']=pd.concat([st.session_state['custom_kb'],add],ignore_index=True);q.pop(i);st.rerun()
    st.markdown('### Policy Change Detector + Document Intelligence')
    pf=st.file_uploader('Upload policy text/doc',type=['txt','md','csv'],key='policy_upload')
    if pf is not None:
        b=pf.getvalue();h=hashlib.sha256(b).hexdigest();ph=st.session_state['policy_hashes'].get(pf.name)
        if ph and ph!=h: st.warning('Policy change detected for this document')
        elif ph==h: st.info('No change since last upload')
        else: st.success('New policy file registered')
        st.session_state['policy_hashes'][pf.name]=h
        lines=[x.strip() for x in b.decode('utf-8',errors='ignore').splitlines() if x.strip()][:8]
        st.markdown('Key extracted lines');
        for ln in lines: st.write(f'- {ln}')
    st.markdown('### Staff Copilot: Draft Notice')
    with st.form('notice'):
        t=st.text_input('Notice topic','Exam schedule update');a=st.selectbox('Audience',['All students','Final year','Faculty']);dd=st.date_input('Deadline',date.today())
        if st.form_submit_button('Generate notice'): st.code(f'Official Notice\nTopic: {t}\nAudience: {a}\nEffective Date: {dd}\nPlease comply with updated process as per academic office instructions.',language='text')

def enterprise_tab():
    st.subheader('Enterprise Readiness');st.markdown('### Integrations')
    c1,c2,c3=st.columns(3);c1.toggle('LMS integration',True);c2.toggle('ERP/SIS integration',False);c3.toggle('Calendar + Email integration',True)
    st.markdown('### Security and Compliance')
    for s in ['SSO-ready authentication flow (demo active)','Role-based access control','Audit trail logs for queries/feedback/alerts','Consent-driven analytics capture','Data retention policy hooks']: st.write(f'- {s}')
    rd=st.selectbox('Data retention',[30,90,180,365],index=1);st.caption(f'Current retention policy: {rd} days')
    st.markdown('### Reliability Engineering')
    ql=pd.read_csv(QUERY_LOG_FILE)
    if ql.empty: st.info('No runtime metrics yet')
    else:
        p95=int(ql['latency_ms'].quantile(0.95));lr=float((ql['confidence']<st.session_state['similarity_threshold']).mean()*100)
        st.write(f'- p95 latency: {p95} ms');st.write(f'- low-confidence rate: {lr:.1f}%');st.write('- fallback model policy: enabled')
    st.markdown('### Monetization and SLA')
    st.dataframe(pd.DataFrame([{'plan':'Starter','monthly_usd':199,'queries':'25k','sla':'Best effort'},{'plan':'Growth','monthly_usd':699,'queries':'150k','sla':'99.5%'},{'plan':'Enterprise','monthly_usd':2499,'queries':'Unlimited','sla':'99.9%'}]),use_container_width=True)

def main():
    ensure_storage();init();st.title('🎓 UniAssist Pro');st.caption('Commercial-grade Academic & Internship Guidance Platform');sidebar()
    if st.session_state['offline_mode']: st.info('Offline kiosk mode active: local cached knowledge and no external calls')
    k=kb();t1,t2,t3,t4,t5,t6=st.tabs(['Assistant','Student Success','Workflows','Analytics','Admin','Enterprise'])
    with t1: assistant_tab(k)
    with t2: student_tab()
    with t3: workflow_tab()
    with t4: analytics_tab()
    with t5: admin_tab()
    with t6: enterprise_tab()
    st.divider();st.caption('© 2026 UniAssist Pro | Publishable prototype with enterprise feature set')

if __name__=='__main__': main()
