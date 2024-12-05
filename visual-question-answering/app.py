import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from intelligent_eye import intelligent_eye

def main():
    st.title('Intelligent EYE')
    # Upload Image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        width_ori, height_ori = image.size

        # Resize the image
        if width_ori > height_ori:
            h_w_ratio = height_ori/width_ori
            WIDTH_NEW = 1024
            HEIGHT_NEW = round(h_w_ratio*WIDTH_NEW)
        else:
            w_h_ratio = width_ori/height_ori
            HEIGHT_NEW = 720
            WIDTH_NEW = round(w_h_ratio*HEIGHT_NEW)
            
        image = image.resize((WIDTH_NEW, HEIGHT_NEW))
        width, height = image.size

        # Draw bounding boxes
        st.write("Select your query objects")
        drawing_mode = "rect"
        stroke_width = 3
        stroke_color = "#00FF00"
        bg_color = "#FFFFFF"
        realtime_update = True
        
        # Canvas component
        canvas_result = st_canvas(
            fill_color="rgba(100, 255, 100, 0.1)",  # Transparent fill
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=image,
            update_streamlit=realtime_update,
            # height=height,
            # width=width,
            drawing_mode=drawing_mode,
            # point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )

        # User question input
        user_input = st.text_input("Enter Your Question")

        # Process the image and user input
        if st.button("Run"):
    
            processed_image, result_text = intelligent_eye(image, canvas_result, user_input)

            # Display processed image and result
            st.image(processed_image, caption='Response', use_column_width=True)
            st.write("Response:", result_text)

if __name__ == "__main__":
    main()
