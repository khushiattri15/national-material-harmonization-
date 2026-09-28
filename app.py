import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="National Material Harmonization", page_icon="🔄", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.hero{padding:22px;border-radius:16px;background:#12304a;margin-bottom:18px}
.hero h1{color:white;margin:0}.hero p{color:#d7e8f5}
</style>
""", unsafe_allow_html=True)

# ---------------- DEMO DATA ----------------
DEMO = [
["IOCL","IOCL-P-001","CENTRIFUGAL PUMP; HEAD 50 M; FLOW 155 M3/HR; MOTOR 45 KW; 1480 RPM; CARBON STEEL; API 610","EA","Pump"],
["BHEL","BHEL-P-101","CENTRIFUGAL PUMP WITH MOTOR; HEAD 50M; FLOW 155 M3/HR; MOTOR 45 KW; SPEED 1480 RPM; CS; API 610","EA","Pump"],
["HPCL","HPCL-P-220","PROCESS CENTRIFUGAL PUMP; HEAD 35 M; FLOW 480 M3/HR; MOTOR 75 KW; 1490 RPM; ASTM A105; API 610","EA","Pump"],
["ONGC","ONGC-P-310","GYPSUM BLEED PUMP WITH MOTOR; HEAD 35M; FLOW 480 M3/HR; MOTOR 75KW; 1490 RPM","EA","Pump"],
["IOCL","IOCL-V-011","GATE VALVE; DN50; CLASS 150; RF; ASTM A105; API 600","EA","Valve"],
["BHEL","BHEL-V-024","GATE VALVE DN 50 CLASS 150 RAISED FACE ASTM A105 API 600","EA","Valve"],
["BPCL","BPCL-V-044","BALL VALVE; 2 INCH; CLASS 300; RF; SS316; API 608","EA","Valve"],
["GAIL","GAIL-P-090","TRANSFER PUMP WITH MOTOR; HEAD 30 M; FLOW 680 M3/HR; MOTOR 110 KW; 1485 RPM","EA","Pump"],
["SAIL","SAIL-P-120","EMERGENCY TRANSFER PUMP WITH MOTOR; HEAD 30M; FLOW 680 M3/HR; MOTOR 110KW; 1485 RPM","EA","Pump"],
["NTPC","NTPC-F-031","SLIP ON FLANGE; DN100; RF; CARBON STEEL; ASTM A105","EA","Flange"],
["CIL","CIL-E-050","90 DEG ELBOW; DN100; BUTT WELD; CARBON STEEL; ASME B16.9","EA","Fitting"],
["CPCL","CPCL-P-077","PRIMARY HYDROCYCLONE FEED TANK PUMP; HEAD 60 M; FLOW 1420 M3/HR; MOTOR IN BHEL SCOPE","EA","Pump"],
]
DEMO_DF = pd.DataFrame(DEMO, columns=["cpse","material_code","original_item_description","unit","category"])

def clean(x):
    return "" if pd.isna(x) else re.sub(r"\s+"," ",str(x)).strip()

def find_col(df, names):
    exact={str(c).strip().lower():c for c in df.columns}
    for n in names:
        if n.lower() in exact:return exact[n.lower()]
    for c in df.columns:
        if any(n.lower() in str(c).lower() for n in names):return c
    return None

def prepare(df):
    d=find_col(df,["original_item_description","material_description","item_description","description","material desc","short text"])
    if not d:return None
    out=df.copy()
    out["__desc"]=out[d].map(clean)
    c=find_col(out,["cpse","company","organization","organisation"])
    code=find_col(out,["material_code","material code","item_code","item code","code"])
    out["__cpse"]=out[c].fillna("Unknown").astype(str) if c else "Unknown"
    out["__code"]=out[code].fillna("").astype(str) if code else [f"MAT-{i+1:04d}" for i in range(len(out))]
    return out

def attrs(desc):
    t=clean(desc).upper()
    r={k:"Not detected" for k in [
        "Material Type","Size","Pressure Class","Connection","Material Grade","Standard",
        "Pump Type","Head","Flow","Motor","Power","Speed","Application","Service","CPSE Scope"
    ]}
    types=["CENTRIFUGAL PUMP","BALL VALVE","GATE VALVE","GLOBE VALVE","CHECK VALVE","BUTTERFLY VALVE","PUMP","PIPE","FLANGE","ELBOW","REDUCER","TEE"]
    for x in types:
        if x in t:r["Material Type"]=x;break

    m=re.search(r"\bDN\s*(\d+)\b",t)
    dnmap={15:"0.5 INCH",20:"0.75 INCH",25:"1 INCH",32:"1.25 INCH",40:"1.5 INCH",50:"2 INCH",65:"2.5 INCH",80:"3 INCH",100:"4 INCH",150:"6 INCH",200:"8 INCH",250:"10 INCH",300:"12 INCH"}
    if m:r["Size"]=dnmap.get(int(m.group(1)),f"DN{m.group(1)}")
    else:
        m=re.search(r"\b(\d+(?:\.\d+)?)\s*(?:IN|INCH)\b",t)
        if m:r["Size"]=m.group(1)+" INCH"

    m=re.search(r"\b(?:CLASS|CL)\s*(\d+)\b",t) or re.search(r"\b(\d+)\s*(?:LB|LBS)\b",t)
    if m:r["Pressure Class"]=m.group(1)

    for label,pat in [("RAISED FACE",r"\bRF\b|RAISED FACE"),("FLAT FACE",r"\bFF\b|FLAT FACE"),("BUTT WELD",r"\bBW\b|BUTT WELD"),("SOCKET WELD",r"\bSW\b|SOCKET WELD"),("THREADED",r"\bTHD\b|THREADED")]:
        if re.search(pat,t):r["Connection"]=label;break

    for label,pat in [("SS304",r"\bSS\s*304\b|\bSS304\b|\b304L?\b"),("SS316",r"\bSS\s*316\b|\bSS316\b|\b316L?\b"),("STAINLESS STEEL",r"STAINLESS STEEL"),("CARBON STEEL",r"CARBON STEEL|\bCS\b"),("CAST IRON",r"CAST IRON|\bCI\b"),("DUCTILE IRON",r"DUCTILE IRON"),("ASTM A105",r"ASTM A105")]:
        if re.search(pat,t):r["Material Grade"]=label;break

    for pat in [r"\bASTM\s*[A-Z]?\s*\d+[A-Z]?\b",r"\bAPI\s*\d+[A-Z]?\b",r"\bASME\s*[A-Z]?\s*\d+[A-Z]?\b",r"\bIS\s*\d+\b",r"\bDIN\s*\d+\b",r"\bISO\s*\d+\b"]:
        m=re.search(pat,t)
        if m:r["Standard"]=re.sub(r"\s+"," ",m.group(0));break

    patterns={
        "Head":r"\bHEAD\s*[:\-]?\s*([\d.]+)\s*M\b",
        "Flow":r"\bFLOW\s*[:\-]?\s*([\d.]+)\s*M3\s*/?\s*HR\b",
        "Motor":r"\bMOTOR\s*[:\-]?\s*([\d.]+)\s*KW\b",
        "Power":r"\b([\d.]+)\s*KW\b",
        "Speed":r"\b([\d.]+)\s*RPM\b",
    }
    for k,p in patterns.items():
        m=re.search(p,t)
        if m:r[k]=m.group(1)+(" M" if k=="Head" else " M3/HR" if k=="Flow" else " KW" if k in ["Motor","Power"] else " RPM")

    if "CENTRIFUGAL" in t:r["Pump Type"]="CENTRIFUGAL"
    elif "RECIPROCATING" in t:r["Pump Type"]="RECIPROCATING"
    elif "SCREW" in t:r["Pump Type"]="SCREW"
    elif r["Material Type"]=="PUMP":r["Pump Type"]="PUMP"

    words=["HYDROCYCLONE","TRANSFER","FEED","DRAIN","BLEED","EMERGENCY","PROCESS","FIRE","SLURRY","COOLING"]
    found=[x for x in words if x in t]
    if found:r["Application"]=", ".join(found)
    if "SLURRY" in t:r["Service"]="SLURRY"
    elif "WATER" in t:r["Service"]="WATER"
    elif "OIL" in t or "CRUDE" in t:r["Service"]="OIL"
    elif "GAS" in t:r["Service"]="GAS"
    if "BHEL SCOPE" in t:r["CPSE Scope"]="BHEL SCOPE"
    return r

def semantic(a,b):
    try:
        v=TfidfVectorizer(ngram_range=(1,2))
        x=v.fit_transform([clean(a).upper(),clean(b).upper()])
        return round(float(cosine_similarity(x[0:1],x[1:2])[0][0])*100,1)
    except Exception:return 0.0

def technical(a,b):
    aa,bb=attrs(a),attrs(b)
    keys=["Material Type","Size","Pressure Class","Connection","Material Grade","Standard","Pump Type","Head","Flow","Motor","Power","Speed","Application","Service"]
    known=match=0; conflicts=[]
    for k in keys:
        if aa[k]!="Not detected" and bb[k]!="Not detected":
            known+=1
            if aa[k].upper()==bb[k].upper():match+=1
            else:conflicts.append((k,aa[k],bb[k]))
    return (round(match/known*100,1) if known else 0.0),conflicts,aa,bb

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ System Controls")
    st.selectbox("Select CPSE",["All CPSEs"])
    upload=st.file_uploader("Upload Material Master",type=["csv","xlsx"])
    st.markdown("---")
    st.markdown("### Prototype Modules")
    st.write("📥 Data ingestion")
    st.write("🤖 AI attribute extraction")
    st.write("🔄 Standardization")
    st.write("🔗 Semantic matching")
    st.write("🧩 Duplicate detection")
    st.write("📊 CPSE analytics")
    st.write("🧠 Explainable review")

if upload:
    try:
        raw=pd.read_excel(upload) if upload.name.lower().endswith(".xlsx") else pd.read_csv(upload)
        df=prepare(raw)
        if df is None:st.error("No description column found. Use original_item_description or description.");st.stop()
        source=upload.name
    except Exception as e:
        st.error(f"Could not read file: {e}");st.stop()
else:
    df=prepare(DEMO_DF);source="Built-in demo dataset"

cpse_list=sorted(df["__cpse"].unique())
with st.sidebar:
    cpse=st.selectbox("Filter dashboard by CPSE",["All CPSEs"]+cpse_list)
work=df if cpse=="All CPSEs" else df[df["__cpse"]==cpse]
if work.empty:st.warning("No materials for this CPSE.");st.stop()

# ---------------- UI ----------------
st.markdown('<div class="hero"><h1>🔄 National Material Harmonization Engine</h1><p>AI-assisted standardization, semantic matching and duplicate detection across CPSE material masters</p></div>',unsafe_allow_html=True)
st.info("SIH 26099 — AI-Driven Standardization and Harmonization of Material Codes Across CPSEs.")

st.header("📊 Dataset Summary")
a,b,c,d=st.columns(4)
a.metric("Materials",len(work));b.metric("Columns",len([x for x in df.columns if not x.startswith("__")]));c.metric("CPSEs",work["__cpse"].nunique());d.metric("Source",source[:20])
with st.expander("Preview material master"):
    shown=[x for x in work.columns if not x.startswith("__")]
    st.dataframe(work[shown].head(15),use_container_width=True,hide_index=True)

st.header("🤖 AI Technical Attribute Extraction")
descs=work["__desc"].tolist()
chosen=st.selectbox("Select a material to analyze",descs)
one=attrs(chosen)
st.markdown("### 🔄 AI Normalization")
st.code(chosen)
keys=["Material Type","Size","Pressure Class","Connection","Material Grade","Standard"]
cols=st.columns(3)
for i,k in enumerate(keys):cols[i%3].metric(k,one[k])
st.markdown("### ⚙️ Category-Specific Attributes")
keys2=["Pump Type","Head","Flow","Motor","Power","Speed","Application","Service","CPSE Scope"]
cols=st.columns(3)
for i,k in enumerate(keys2):cols[i%3].metric(k,one[k])

st.header("🔗 AI Material Match Explorer")
if len(descs)>=2:
    ia=st.selectbox("Material A",range(len(descs)),format_func=lambda i:descs[i][:110],key="A")
    ib=st.selectbox("Material B",range(len(descs)),index=1,format_func=lambda i:descs[i][:110],key="B")
    sa,sb=descs[ia],descs[ib]
    sem=semantic(sa,sb)
    tech,conf,aa,bb=technical(sa,sb)
    overall=round(.55*sem+.45*tech,1)
    x,y,z=st.columns(3);x.metric("Semantic Similarity",f"{sem}%");y.metric("Technical Match",f"{tech}%");z.metric("Overall Score",f"{overall}%")
    if overall>=85:st.success("🟢 HIGH SIMILARITY — candidate for harmonization review")
    elif overall>=65:st.warning("🟡 MEDIUM SIMILARITY — human review recommended")
    else:st.error("🔴 LOW SIMILARITY — likely different materials")
    rows=[]
    for k in ["Material Type","Size","Pressure Class","Connection","Material Grade","Standard","Pump Type","Head","Flow","Motor","Power","Speed","Application","Service"]:
        av,bv=aa[k],bb[k]
        rows.append([k,av,bv,"MATCH" if av!="Not detected" and bv!="Not detected" and av.upper()==bv.upper() else "REVIEW"])
    st.subheader("🔍 Technical Attribute Comparison")
    st.dataframe(pd.DataFrame(rows,columns=["Attribute","Material A","Material B","Status"]),use_container_width=True,hide_index=True)
    if conf:
        st.warning("Technical differences detected:")
        for k,av,bv in conf:st.write(f"**{k}:** A = `{av}` | B = `{bv}`")
    else:st.success("✅ No detected technical conflicts.")

st.header("🧩 Duplicate & Harmonization Candidate Detection")
threshold=st.slider("Similarity threshold",50,95,75,5)
pairs=[]
for i in range(len(work)):
    for j in range(i+1,len(work)):
        sem=semantic(work.iloc[i]["__desc"],work.iloc[j]["__desc"])
        if sem>=threshold:
            tech,_,_,_=technical(work.iloc[i]["__desc"],work.iloc[j]["__desc"])
            overall=round(.55*sem+.45*tech,1)
            pairs.append([work.iloc[i]["__cpse"],work.iloc[i]["__code"],work.iloc[j]["__cpse"],work.iloc[j]["__code"],sem,tech,overall])
if pairs:
    p=pd.DataFrame(pairs,columns=["CPSE A","Code A","CPSE B","Code B","Semantic %","Technical %","Overall %"]).sort_values("Overall %",ascending=False)
    st.dataframe(p.head(30),use_container_width=True,hide_index=True)
    st.caption("AI-assisted candidates only. Final equivalence requires authorized technical/procurement validation.")
else:st.info("No candidate pairs crossed the selected threshold.")

st.header("🏢 Cross-CPSE Analytics")
summary=df.groupby("__cpse").size().reset_index(name="Material Count").rename(columns={"__cpse":"CPSE"})
st.dataframe(summary.sort_values("Material Count",ascending=False),use_container_width=True,hide_index=True)

st.header("📥 Export")
out=work[[x for x in work.columns if not x.startswith("__")]].copy()
st.download_button("⬇️ Download filtered material master",out.to_csv(index=False).encode(),file_name="harmonized_material_master.csv",mime="text/csv")
st.markdown("---")
st.caption("SIH 26099 prototype • TF-IDF semantic similarity + explainable technical attribute extraction")
