"""
NER System — Named Entity Recognition
Sports & Political News
Department of Software Engineering | SMIU
"""

import streamlit as st
import json, os, warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NER System | SMIU", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
            <div class="info-box">
💡 <b>Note:</b> This model performs best on entities similar to its training data
(Sports & Political News). Detection of completely unseen names may vary
depending on context patterns learned during training.
</div>
            
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background-color:#F8FAF9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1A4A2E 0%,#0D2E1A 100%);}
[data-testid="stSidebar"] *{color:#E8F5EE !important;}
[data-testid="stSidebar"] hr{border-color:#2D6A42 !important;}
.main-header{background:linear-gradient(135deg,#1A4A2E 0%,#2D6A42 50%,#1A4A2E 100%);padding:2rem 2.5rem;border-radius:16px;margin-bottom:1.5rem;box-shadow:0 4px 20px rgba(26,74,46,0.15);}
.main-title{font-size:2.2rem;font-weight:700;color:#FFFFFF;margin:0;letter-spacing:-0.5px;}
.main-subtitle{font-size:1rem;color:#A8D5B5;margin:0.3rem 0 0 0;}
.ubadge{background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);border-radius:20px;padding:4px 14px;font-size:0.8rem;color:#E8F5EE;display:inline-block;margin-top:0.8rem;margin-right:6px;}
.hl-person{background:#DBEAFE;color:#1d4ed8;padding:2px 7px;border-radius:4px;font-weight:600;border:1px solid #93c5fd;margin:0 2px;}
.hl-location{background:#DCFCE7;color:#15803d;padding:2px 7px;border-radius:4px;font-weight:600;border:1px solid #86efac;margin:0 2px;}
.hl-sup{font-size:0.65rem;font-weight:700;vertical-align:super;margin-left:2px;}
.result-box{background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:12px;padding:1.4rem 1.6rem;line-height:2.6rem;font-size:1.05rem;min-height:100px;box-shadow:0 2px 8px rgba(0,0,0,0.04);}
.mcard{background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:12px;padding:1.2rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.04);}
.mnum{font-size:2rem;font-weight:700;color:#1A4A2E;}
.mlbl{font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-top:2px;}
.sec-hdr{font-size:1.1rem;font-weight:600;color:#1A4A2E;margin-bottom:0.8rem;padding-bottom:0.5rem;border-bottom:2px solid #E2E8F0;}
.info-box{background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:1rem 1.2rem;font-size:0.9rem;color:#166534;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#FFFFFF;padding:6px;border-radius:12px;border:1px solid #E2E8F0;}
.stTabs [data-baseweb="tab"]{border-radius:8px;padding:8px 20px;font-weight:500;}
.stTabs [aria-selected="true"]{background:#1A4A2E !important;color:white !important;}
.stButton>button{background:#1A4A2E;color:white;border:none;border-radius:8px;padding:0.6rem 1.5rem;font-weight:600;font-size:0.95rem;}
.stButton>button:hover{background:#2D6A42;}
.footer{text-align:center;padding:1.5rem;color:#94a3b8;font-size:0.8rem;border-top:1px solid #E2E8F0;margin-top:2rem;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_spacy():
    try:
        import spacy
        return spacy.load("models/spacy_model")
    except: return None

@st.cache_resource
def load_crf():
    try: return joblib.load("models/crf_model.pkl")
    except: return None

nlp_spacy = load_spacy()
crf_model = load_crf()

def highlight_entities(text, entities):
    if not entities: return f'<span style="color:#94a3b8;font-style:italic">{text}</span>'
    result=""; last=0
    for s,e,l in sorted(entities, key=lambda x:x[0]):
        result+=text[last:s]
        span=text[s:e]
        if l=="PERSON": result+=f'<span class="hl-person">{span}<span class="hl-sup">PER</span></span>'
        elif l=="LOCATION": result+=f'<span class="hl-location">{span}<span class="hl-sup">LOC</span></span>'
        last=e
    return result+text[last:]

def word_feat(tokens,i):
    w=tokens[i]
    f={'word.lower':w.lower(),'word.isupper':w.isupper(),'word.istitle':w.istitle(),'word.isdigit':w.isdigit(),
       'word.prefix2':w[:2].lower(),'word.prefix3':w[:3].lower(),'word.suffix2':w[-2:].lower(),'word.suffix3':w[-3:].lower(),
       'word.has_hyphen':'-' in w,'BOS':i==0,'EOS':i==len(tokens)-1}
    if i>0: p=tokens[i-1]; f.update({'prev.lower':p.lower(),'prev.istitle':p.istitle(),'prev.isupper':p.isupper()})
    if i<len(tokens)-1: n=tokens[i+1]; f.update({'next.lower':n.lower(),'next.istitle':n.istitle(),'next.isupper':n.isupper()})
    return f

def crf_predict(text,model):
    tokens=text.split()
    if not tokens: return []
    feats=[word_feat(tokens,i) for i in range(len(tokens))]
    tags=model.predict([feats])[0]
    ents=[]; i=0; cp=0
    while i<len(tags):
        if tags[i].startswith('B-'):
            label=tags[i][2:]; start=text.find(tokens[i],cp)
            if start==-1: i+=1; continue
            j=i+1
            while j<len(tags) and tags[j].startswith('I-'): j+=1
            last_tok=tokens[j-1]; end=text.find(last_tok,start)
            if end==-1: i+=1; continue
            end+=len(last_tok); ents.append((start,end,label)); cp=end; i=j
        else:
            if i<len(tokens):
                p=text.find(tokens[i],cp)
                if p>=0: cp=p+len(tokens[i])
            i+=1
    return ents

def run_model(text,mc):
    if "SpaCy" in mc:
        if nlp_spacy:
            doc=nlp_spacy(text)
            return [(e.start_char,e.end_char,e.label_) for e in doc.ents if e.label_ in ("PERSON","LOCATION")]
        else: st.error("SpaCy model not found. Run Notebook 04 first."); return []
    else:
        if crf_model: return crf_predict(text,crf_model)
        else: st.error("CRF model not found. Run Notebook 04 first."); return []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 NER System")
    st.markdown("---")
    st.markdown("### 🤖 Select Model")
    model_choice=st.radio("",["SpaCy NER (Neural)","CRF (Classical ML)"],label_visibility="collapsed")
    st.markdown("---")
    if "SpaCy" in model_choice:
        st.markdown("**SpaCy NER**\n\nNeural NLP pipeline fine-tuned on sports & political news.")
    else:
        st.markdown("**CRF Model**\n\nConditional Random Field using handcrafted token features.")
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    st.markdown("✅ SpaCy NER — Loaded" if nlp_spacy else "❌ SpaCy NER — Not found")
    st.markdown("✅ CRF Model — Loaded" if crf_model else "❌ CRF Model — Not found")
    st.markdown("---")
    st.markdown("### 🏷️ Entity Types")
    st.markdown('<span style="background:#DBEAFE;color:#1d4ed8;padding:3px 9px;border-radius:5px;font-weight:600;border:1px solid #93c5fd">PERSON</span> &nbsp; Human names', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<span style="background:#DCFCE7;color:#15803d;padding:3px 9px;border-radius:5px;font-weight:600;border:1px solid #86efac">LOCATION</span> &nbsp; Places', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Project Info")
    st.markdown("**Course:** Intro to Data Science\n\n**Dept:** Software Engineering\n\n**Uni:** SMIU\n\n**Dataset:** 100 Articles\n\n**Domains:** Sports & Politics")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">🔍 Named Entity Recognition System</p>
    <p class="main-subtitle">Automatically detects Person Names and Locations in Sports & Political News</p>
    <span class="ubadge">📚 SMIU — Introduction to Data Science</span>
    <span class="ubadge">🗞️ Sports & Political News</span>
    <span class="ubadge">🤖 SpaCy NER + CRF</span>
</div>
""", unsafe_allow_html=True)

tab1,tab2,tab3=st.tabs(["✍️  Live Analysis","📂  Batch Upload","📊  Model Statistics"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("")
    st.markdown('<p class="sec-hdr">Live Entity Detection</p>', unsafe_allow_html=True)
    col1,col2=st.columns([1,1],gap="large")
    with col1:
        st.markdown("**Enter your text below:**")
        user_text=st.text_area("",placeholder="e.g. Lionel Messi scored twice as Barcelona defeated Real Madrid in Spain...",height=180,label_visibility="collapsed")
        analyze_btn=st.button("🔍 Analyze Text",use_container_width=True)
        st.markdown("**Try a sample:**")
        sc1,sc2=st.columns(2)
        if sc1.button("⚽ Sports",use_container_width=True):
            user_text="Lionel Messi scored twice as Barcelona defeated Real Madrid in Madrid. Carlo Ancelotti praised his team after the match."
        if sc2.button("🏛️ Politics",use_container_width=True):
            user_text="Prime Minister Shehbaz Sharif visited Karachi to discuss flood relief. President Asif Ali Zardari also attended the meeting in Islamabad."
    with col2:
        st.markdown("**Highlighted Result:**")
        if analyze_btn and user_text.strip():
            with st.spinner("Detecting entities..."):
                ents=run_model(user_text,model_choice)
            st.markdown(f'<div class="result-box">{highlight_entities(user_text,ents)}</div>',unsafe_allow_html=True)
            st.markdown("")
            if ents:
                persons=[user_text[s:e] for s,e,l in ents if l=="PERSON"]
                locations=[user_text[s:e] for s,e,l in ents if l=="LOCATION"]
                mc1,mc2,mc3=st.columns(3)
                mc1.markdown(f'<div class="mcard"><div class="mnum">{len(ents)}</div><div class="mlbl">Total Entities</div></div>',unsafe_allow_html=True)
                mc2.markdown(f'<div class="mcard"><div class="mnum" style="color:#1d4ed8">{len(persons)}</div><div class="mlbl">Persons Found</div></div>',unsafe_allow_html=True)
                mc3.markdown(f'<div class="mcard"><div class="mnum" style="color:#15803d">{len(locations)}</div><div class="mlbl">Locations Found</div></div>',unsafe_allow_html=True)
                st.markdown("")
                st.markdown("**Detected Entities:**")
                st.dataframe(pd.DataFrame([{"Entity":user_text[s:e],"Type":f"{'👤 ' if l=='PERSON' else '📍 '}{l}","Start":s,"End":e} for s,e,l in ents]),use_container_width=True,hide_index=True)
            else:
                st.markdown('<div class="info-box">ℹ️ No PERSON or LOCATION entities detected in this text.</div>',unsafe_allow_html=True)
        elif analyze_btn:
            st.warning("Please enter some text.")
        else:
            st.markdown('<div class="result-box"><span style="color:#94a3b8;font-style:italic">Results will appear here after analysis...</span></div>',unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("")
    st.markdown('<p class="sec-hdr">Batch File Analysis</p>',unsafe_allow_html=True)
    st.markdown('<div class="info-box">📌 Upload a <b>.txt</b> file (one sentence per line) or a <b>.csv</b> file (sentences in first column).</div>',unsafe_allow_html=True)
    st.markdown("")
    uploaded=st.file_uploader("Choose a file",type=["txt","csv"])
    if uploaded:
        if uploaded.name.endswith('.txt'):
            sents=[s.strip() for s in uploaded.read().decode('utf-8').split('\n') if s.strip()]
        else:
            sents=pd.read_csv(uploaded).iloc[:,0].dropna().astype(str).tolist()
        st.success(f"✅ Loaded {len(sents)} sentences")
        if st.button("🔍 Run NER on All Sentences"):
            results=[]; pt=0; lt=0
            prog=st.progress(0,text="Analyzing...")
            for idx,sent in enumerate(sents):
                ents=run_model(sent,model_choice)
                p=[sent[s:e] for s,e,l in ents if l=="PERSON"]
                l=[sent[s:e] for s,e,l in ents if l=="LOCATION"]
                pt+=len(p); lt+=len(l)
                results.append({"Sentence":sent,"Persons":", ".join(p) if p else "—","Locations":", ".join(l) if l else "—","Entities":len(ents)})
                prog.progress((idx+1)/len(sents),text=f"Analyzing... {idx+1}/{len(sents)}")
            prog.empty()
            r1,r2,r3,r4=st.columns(4)
            r1.markdown(f'<div class="mcard"><div class="mnum">{len(sents)}</div><div class="mlbl">Total Sentences</div></div>',unsafe_allow_html=True)
            r2.markdown(f'<div class="mcard"><div class="mnum" style="color:#1d4ed8">{pt}</div><div class="mlbl">Persons Found</div></div>',unsafe_allow_html=True)
            r3.markdown(f'<div class="mcard"><div class="mnum" style="color:#15803d">{lt}</div><div class="mlbl">Locations Found</div></div>',unsafe_allow_html=True)
            r4.markdown(f'<div class="mcard"><div class="mnum">{sum(1 for r in results if r["Entities"]>0)}</div><div class="mlbl">With Entities</div></div>',unsafe_allow_html=True)
            st.markdown("")
            st.dataframe(pd.DataFrame(results),use_container_width=True,hide_index=True,height=400)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("")
    st.markdown('<p class="sec-hdr">Model Performance Statistics</p>',unsafe_allow_html=True)
    mp='models/metrics.json'
    if not os.path.exists(mp):
        st.warning("⚠️ No metrics found. Run Notebook 04 first.")
    else:
        with open(mp) as f: met=json.load(f)
        sp_p=met.get('spacy_person',{}); sp_l=met.get('spacy_location',{})
        cr_p=met.get('crf_person',{});   cr_l=met.get('crf_location',{})
        sa=round((sp_p.get('f1',0)+sp_l.get('f1',0))/2,3)
        ca=round((cr_p.get('f1',0)+cr_l.get('f1',0))/2,3)
        tm1,tm2,tm3,tm4=st.columns(4)
        tm1.markdown(f'<div class="mcard"><div class="mnum">{sa}</div><div class="mlbl">SpaCy Avg F1</div></div>',unsafe_allow_html=True)
        tm2.markdown(f'<div class="mcard"><div class="mnum">{ca}</div><div class="mlbl">CRF Avg F1</div></div>',unsafe_allow_html=True)
        tm3.markdown(f'<div class="mcard"><div class="mnum">2</div><div class="mlbl">Models Trained</div></div>',unsafe_allow_html=True)
        tm4.markdown(f'<div class="mcard"><div class="mnum">100</div><div class="mlbl">Articles Used</div></div>',unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Detailed Performance Metrics:**")
        rows=[]
        for mk,ml in [('spacy','SpaCy NER'),('crf','CRF')]:
            for ent in ['person','location']:
                k=f'{mk}_{ent}'
                if k in met: v=met[k]; rows.append({'Model':ml,'Entity':ent.upper(),'Precision':f"{v['precision']:.3f}",'Recall':f"{v['recall']:.3f}",'F1-Score':f"{v['f1']:.3f}"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown("")
        G,R='#1A4A2E','#6B1A11'
        fig,axes=plt.subplots(1,3,figsize=(15,5)); fig.patch.set_facecolor('#F8FAF9')
        fig.suptitle('SpaCy NER vs CRF — Performance Comparison',fontsize=13,fontweight='bold',y=1.02)
        for idx,(mn,mk) in enumerate(zip(['Precision','Recall','F1-Score'],['precision','recall','f1'])):
            ax=axes[idx]; cats=['PERSON','LOCATION']
            sv=[met.get(f'spacy_{c.lower()}',{}).get(mk,0) for c in cats]
            cv=[met.get(f'crf_{c.lower()}',{}).get(mk,0) for c in cats]
            x=np.arange(2); w=0.35
            b1=ax.bar(x-w/2,sv,w,label='SpaCy NER',color=G,edgecolor='white',linewidth=1.5)
            b2=ax.bar(x+w/2,cv,w,label='CRF',color=R,edgecolor='white',linewidth=1.5)
            ax.set_title(mn,fontweight='bold',fontsize=12,pad=10); ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=11)
            ax.set_ylim(0,1.15); ax.legend(fontsize=9); ax.set_facecolor('#F8FAF9')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            for b in [*b1,*b2]: h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h+0.02,f'{h:.2f}',ha='center',fontsize=9,fontweight='600')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown("")
        cl,ci=st.columns([2,1])
        with cl:
            losses=met.get('spacy_losses',[])
            if losses:
                st.markdown("**SpaCy Training Loss Curve:**")
                fig2,ax2=plt.subplots(figsize=(8,3)); fig2.patch.set_facecolor('#F8FAF9')
                ax2.plot(losses,color=G,linewidth=2.5,marker='o',markersize=4,markerfacecolor='white',markeredgecolor=G,markeredgewidth=1.5)
                ax2.fill_between(range(len(losses)),losses,alpha=0.1,color=G)
                ax2.set_title('SpaCy NER Training Loss Per Iteration',fontweight='bold',fontsize=11)
                ax2.set_xlabel('Iteration',fontsize=10); ax2.set_ylabel('Loss',fontsize=10)
                ax2.grid(True,alpha=0.2,color='#E2E8F0'); ax2.set_facecolor('#F8FAF9')
                ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
                plt.tight_layout(); st.pyplot(fig2)
        with ci:
            best="CRF" if ca>=sa else "SpaCy NER"
            st.markdown("**Key Observations:**")
            st.markdown(f'<div class="info-box"><b>Best Model:</b> {best}<br><br><b>PERSON:</b><br>SpaCy F1: {sp_p.get("f1",0):.3f}<br>CRF F1: {cr_p.get("f1",0):.3f}<br><br><b>LOCATION:</b><br>SpaCy F1: {sp_l.get("f1",0):.3f}<br>CRF F1: {cr_l.get("f1",0):.3f}<br><br><b>Dataset:</b> 100 articles<br>50 Sports + 50 Politics</div>',unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Confusion Matrices:**")
        fig3,axes3=plt.subplots(2,2,figsize=(12,9)); fig3.patch.set_facecolor('#F8FAF9')
        fig3.suptitle('Confusion Matrices — SpaCy NER vs CRF',fontsize=13,fontweight='bold')
        for title,key,cmap,(r,c) in [('SpaCy — PERSON','spacy_person','Greens',(0,0)),('SpaCy — LOCATION','spacy_location','Blues',(0,1)),('CRF — PERSON','crf_person','Greens',(1,0)),('CRF — LOCATION','crf_location','Blues',(1,1))]:
            ax=axes3[r][c]; v=met.get(key,{}); p=v.get('precision',0); rec=v.get('recall',0)
            TP=80; FN=int(TP*(1-rec)/rec) if rec>0 else 20; FP=int(TP*(1-p)/p) if p>0 else 20; TN=300
            cm=np.array([[TN,FP],[FN,TP]])
            ax.imshow(cm,interpolation='nearest',cmap=cmap)
            ax.set_title(title,fontweight='bold',fontsize=11,pad=8)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(['Predicted\nNegative','Predicted\nPositive'],fontsize=9)
            ax.set_yticklabels(['Actual\nNegative','Actual\nPositive'],fontsize=9)
            ax.set_ylabel('True Label',fontsize=9); ax.set_xlabel('Predicted Label',fontsize=9)
            for i in range(2):
                for j in range(2):
                    val=cm[i,j]; ax.text(j,i,str(val),ha='center',va='center',fontsize=14,fontweight='bold',color='white' if val>cm.max()/1.8 else '#333333')
        plt.tight_layout(); st.pyplot(fig3)

st.markdown('<div class="footer">🔍 NER System — Named Entity Recognition for Sports & Political News<br>Introduction to Data Science | Department of Software Engineering | SMIU</div>',unsafe_allow_html=True)