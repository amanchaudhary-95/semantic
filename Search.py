import streamlit as st
from streamlit_pdf_reader import pdf_reader
import helper_fxn as hf
from PIL import Image

img = Image.open('./Logo.png')
icon = Image.open('./favi.png')

def read_pdf(file):
    with open('./Processed/'+file, 'rb') as f:
        temp = f.read()
    return temp

@st.dialog('PDF Preview', width='large')
def open_dialog(file):
    with st.container(height=460):
        pdf_reader('./Processed/'+file)
    

st.set_page_config(page_title='Sementic Search Engine', page_icon=icon, layout="wide", initial_sidebar_state="collapsed",
        menu_items={'About': f"## Travel Portal\n**Developed by** : DCoE Team\n**Version** : 3.0"})
st.markdown('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.markdown("""<style> header {visibility: hidden;} </style>""", unsafe_allow_html=True)

if 'download' not in st.session_state:
    st.session_state['models'] = {'T5':'t5', 'DISTILBERT':'distilbert', 'BERT':'bert', 'MiniLM':'minilm'}
    st.session_state['download'] = False
    st.session_state['popup'] = False

if st.session_state['popup'] == False:
    hf.popup()
    st.session_state['popup'] = True


c1, c2, c4, c3 = st.columns([0.42, 2, 0.4, 0.4], gap='small', vertical_alignment='bottom')
c1.image(img, width=150)
c2.subheader(f'**Semantic Search Engine**', divider='rainbow')
choose_model = c4.selectbox('Choose Model', st.session_state['models'], index=0)
popover = c3.popover(':material/logout: **Logout**', use_container_width=True, help='User details and Logout from the application')  
popover.write(f'**👤Guest**')

form = st.form('form')
col1, col2 = form.columns([1, 0.04], gap='small', vertical_alignment='bottom') 
query = col1.text_input("Search box", placeholder='Enter your query here', label_visibility='collapsed')
if col2.form_submit_button(':material/search:', type='primary') or st.session_state['download']:
    if query != "":
        st.session_state['download'] = True
        model = hf.load_model(choose_model)
        index = hf.read_faiss(choose_model)
        query_vector = model.encode([query])

        results, net_t = hf.search(index, query_vector, top_k=10, choose_model=choose_model)
        st.caption(f':gray-background[:blue[**{len(results)}**] Result(s) in :green[**{net_t:.4f}**] second(s)]')
        # st.info(f'Results in second(s): {net_t:.4f}')
        for i, r in enumerate(results):
            exp = st.container(border=True)
            c1,c2, c3 = exp.columns([1, 0.13, 0.15])
            c1.write(f"**File Name: :violet[{r['title']}]**")
            exp.markdown(f':blue[Content] : {r["content"].capitalize()}', unsafe_allow_html=True)  
            if c2.button(':material/picture_as_pdf: View File', key=f"b_{r['title']}{i}"):
                open_dialog(r['title'])
            c3.download_button(':material/download: Download', file_name=r['title'], data=read_pdf(r['title']), key=f"{r['title']}{i}")

    else:
        st.warning('Enter Search Query')
        st.session_state['download'] = False