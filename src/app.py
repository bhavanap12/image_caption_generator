import streamlit as st
from PIL import Image
from sample import main
from build_vocab import Vocabulary

st.title(":blue[Image Caption Generator] :writing_hand:")
st.caption(":gray[RNN trained on coco2014 dataset]")
st.subheader("", divider='rainbow')
st.subheader("Please upload an image file")
file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.write("")
    # st.button("Generate caption")
else:
    image = Image.open(file)
    #, use_column_width=True)
    
    st.image(image)
    if st.button(":violet[Generate caption]"):
        caption = main(file)
        st.subheader(":green["+ caption +"]")
    # st.image(image, caption)
    
#     predictions = import_and_predict(image, model)
#     score = tf.nn.softmax(predictions[0])
#     st.write(prediction)
#     st.write(score)
#     print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )