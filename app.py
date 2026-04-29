import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore")

genai.configure(api_key="AIzaSyDLwCXbGtlGHhhmoUGMsdcQz2DIVhXV1Ww")
model_gemini = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

MODEL_PATH = "brain_tumor_xception.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = ["U thần kinh đệm (Glioma)", "U màng não (Meningioma)", "Không có khối u", "U tuyến yên (Pituitary)"]
CLASS_NAMES_EN = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def preprocess(img):
    img = img.resize((299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def doctor_reply(pred_class, prob_dict):
    prompt = f"""
    Bạn là một bác sĩ chuyên khoa chẩn đoán MRI não.
    Kết quả quét MRI của bệnh nhân:
    - Loại khối u dự đoán: {pred_class}
    - Xác suất của mô hình: {prob_dict}
    
    Vui lòng:
    1. Giải thích khối u này có nghĩa gì bằng ngôn ngữ đơn giản, dễ hiểu.
    2. Đưa ra lời khuyên về các bước y tế tiếp theo.
    
    Trả lời BẰNG TIẾNG VIỆT.
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi khi tạo phản hồi: {str(e)}"

st.title("Chatbot Chẩn đoán Khối u Não từ Ảnh MRI")

# Khởi tạo session state cho lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None
if "prob_dict" not in st.session_state:
    st.session_state.prob_dict = None

uploaded_file = st.file_uploader("Tải lên hình ảnh MRI", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Hình ảnh MRI đã tải lên", use_container_width=True)
    
    with st.spinner("AI đang phân tích ảnh chụp MRI..."):
        x = preprocess(img)
        preds = model.predict(x)[0]
        pred_index = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[pred_index]
        predicted_class_en = CLASS_NAMES_EN[pred_index]
        prob_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    
    st.success(f"Kết quả dự đoán: **{predicted_class}**")
    st.write("Xác suất:", prob_dict)
    
    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.prob_dict = prob_dict
    
    if not st.session_state.chat_history:
        with st.spinner("Bác sĩ đang viết lời khuyên cá nhân hóa cho bạn..."):
            advice = doctor_reply(predicted_class, prob_dict)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": advice
        })

chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**Bạn:** {message['content']}")
        else:
            st.markdown(f"**Bác sĩ:** {message['content']}")
        st.markdown("---")

st.subheader("Đặt câu hỏi cho bác sĩ")

user_input = st.text_area("Nhập câu hỏi của bạn ở đây:", height=100)

if st.button("Gửi", type="primary"):
    if user_input.strip():
        user_message = user_input.strip()
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_message
        })
        
        with st.spinner("Bác sĩ đang trả lời..."):
            if st.session_state.prediction_made:
                context_prompt = f"""
                Bạn là một bác sĩ y khoa. Bệnh nhân đã được chẩn đoán:
                - Loại khối u: {st.session_state.predicted_class}
                - Xác suất: {st.session_state.prob_dict}
                
                Câu hỏi của bệnh nhân: {user_input}
                
                Vui lòng cung cấp câu trả lời hữu ích, đồng cảm và chuyên nghiệp.
                Trả lời BẰNG TIẾNG VIỆT.
                """
            else:
                context_prompt = f"""
                Bạn là bác sĩ chuyên khoa về khối u não.
                
                Câu hỏi của bệnh nhân: {user_input}
                
                Vui lòng cung cấp câu trả lời hữu ích, đồng cảm và chuyên nghiệp.
                Trả lời BẰNG TIẾNG VIỆT.
                """
            
            try:
                response = model_gemini.generate_content(context_prompt)
                doctor_response = response.text
            except Exception as e:
                doctor_response = f"Xin lỗi, hiện tại tôi gặp khó khăn trong việc trả lời. Lỗi: {str(e)}"
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": doctor_response
        })
        st.rerun()
    else:
        st.warning("Vui lòng nhập tin nhắn hoặc đính kèm hình ảnh.")

