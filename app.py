import streamlit as st
import streamlit.components.v1 as stc
import analysis as a
import ml as m
import cv2

st.title('The AI Solution')


def load_img(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main():
    menu = ['Home', 'Analysis', 'Machine Learning', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.header("Your Machine Learning Solution")
        st.image(load_img('data_visualization.jpg'), use_column_width=True)

    elif choice == 'Analysis':
        st.header('Data Analysis')
        a.BI()

    elif choice == 'Machine Learning':
        st.header(choice)
        m.ML()

    else:
        st.header(choice)


if __name__ == '__main__':
    main()
