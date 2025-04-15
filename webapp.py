import streamlit as st
import re
import sqlite3 
import pickle
import bz2
import pandas as pd
import numpy as np
st.set_page_config(page_title="Multi-Syndrome Classification", page_icon="fevicon.webp", layout="centered", initial_sidebar_state="auto", menu_items=None)

conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(FirstName TEXT,LastName TEXT,Mobile TEXT,City TEXT,Email TEXT,password TEXT,Cpassword TEXT)')
def add_userdata(FirstName,LastName,Mobile,City,Email,password,Cpassword):
    c.execute('INSERT INTO userstable(FirstName,LastName,Mobile,City,Email,password,Cpassword) VALUES (?,?,?,?,?,?,?)',(FirstName,LastName,Mobile,City,Email,password,Cpassword))
    conn.commit()
def login_user(Email,password):
    c.execute('SELECT * FROM userstable WHERE Email =? AND password = ?',(Email,password))
    data = c.fetchall()
    return data
def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data
def delete_user(Email):
    c.execute("DELETE FROM userstable WHERE Email="+"'"+Email+"'")
    conn.commit()


st.title("Multi-Syndrome Classification")



menu = ["Home","Login","SignUp"]
choice = st.sidebar.selectbox("Menu",menu)

if choice=="Home":
    st.markdown(
        """
        <p align="justify">
        <b style="color:black">Multi-Syndrome diseases as Diabetes, Anemia, Thalassemia, Heart illnesses, Thrombocytopenia, and Health is important for the diagnosis and treatment within the sphere of healthcare. This paper offers a new approach to the categorization of syndromes employing machine learning approaches. In machine learning Procedure, the Random Forest Algorithm, and the Extra Trees modules are used to improve the classifier’s speed and ability to avoid noise data. Feature selection techniques are employed in extracting features from different medical data sources and thus enhancing discriminant property of made models. A cost-benefit analysis is therefore commenced, whereby the performance of the combined ensemble models is compared with that of the conventional single-model techniques. Numerous studies that examine medical datasets show that our approach to ensemble learning significantly outperforms other frameworks for the investigation of a range of conditions. This study greatly enriches the knowledge of syndrome classification and provides a reliable adjacent syntactic classifier for clinicians and scholars who are involved in various types of disorders.</b>
        </p>
        """
        ,unsafe_allow_html=True)
    
if 'reset' not in st.session_state:
    st.session_state.reset = False

if choice == "Login":
    Email = st.sidebar.text_input("Email")
    Password = st.sidebar.text_input("Password", type="password")
    b1 = st.sidebar.checkbox("Login")

    if b1:
        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.fullmatch(regex, Email):
            create_usertable()
            if Email == 'a@a.com' and Password == '123':
                st.success(f"Logged In as Admin")
                Email = st.text_input("Delete Email")
                if st.button('Delete'):
                    delete_user(Email)
                user_result = view_all_users()
                clean_db = pd.DataFrame(user_result, columns=["FirstName", "LastName", "Mobile", "City", "Email", "password", "Cpassword"])
                st.dataframe(clean_db)
            else:
                result = login_user(Email, Password)
                if result:
                    st.success(f"Logged In as {Email}")
                    menu2 = ["K-Nearest Neighbors","Decision Tree", "Random Forest","Naive Bayes","ExtraTreesClassifier"]
                    choice2 = st.selectbox("Select ML", menu2)

                    sfile1 = bz2.BZ2File('features.pkl', 'r')
                    selected_features = pickle.load(sfile1)

                    # Initialize session state for storing the feature choices (default to "False")
                    if 'choices' not in st.session_state:
                        st.session_state.choices = ["False"] * len(selected_features)  # Default all to "False"

                    # Loop through selected features and create selectboxes
                    for i, feature in enumerate(selected_features):
                        key_name = f"feature_{i}"

                        # If reset is triggered, set all choices to "False"
                        if st.session_state.reset:
                            st.session_state.choices[i] = "False"  # Reset to False

                        # Ensure that choices are valid values ("False" or "True")
                        val = st.selectbox(feature, ["False", "True"], key=key_name, index=["False", "True"].index(st.session_state.choices[i]))
                        st.session_state.choices[i] = val  # Update the session state with the selected value

                    

                    b2 = st.button("Predict")
                    # Add reset button to reset all selections to False
                    if st.button("Reset All Selections"):
                        st.session_state.reset = True
                    else:
                        st.session_state.reset = False

                    sfile = bz2.BZ2File('model.pkl', 'r')
                    model = pickle.load(sfile)
                    tdata = [1 if choice == "True" else 0 for choice in st.session_state.choices]

                    df = pd.read_csv("Disease precaution.csv")
                    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
                    diseases = ['drug reaction', 'allergy', 'common cold', 'chickenpox',
                                'neonatal jaundice', 'pneumonia', 'infectious gastroenteritis']
                    df["Disease"] = df["Disease"].str.replace("chicken pox", "chickenpox")
                    df["Disease"] = df["Disease"].str.replace("jaundice", "neonatal jaundice")
                    df["Disease"] = df["Disease"].str.replace("gastroenteritis", "infectious gastroenteritis")
                    df = df[df['Disease'].isin(diseases)]
                    df = df.fillna("")
                    df['Precautions'] = df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].apply(lambda x: ', '.join(x), axis=1)
                    df.drop(columns=['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'], inplace=True)
                    df.reset_index(inplace=True, drop=True)

                    if b2:
                        if len(np.unique(tdata)) == 1:
                            if np.unique(tdata)[0] == 1:
                                st.success("Please Contact Nearest Doctor")
                            else:
                                st.success("You are healthy")
                        else:
                            index = ["K-Nearest Neighbors", "SVM", "Decision Tree", "Random Forest",
                                     "Naive Bayes", "ExtraTreesClassifier", "VotingClassifier"].index(choice2)
                            test_prediction = model[4].predict([tdata])
                            query = test_prediction[0]
                            score = np.amax(model[4].predict_proba([tdata]))
                            st.success(query)
                            st.success(f"Probability: {score}")
                            st.success(df[df['Disease'] == query]["Precautions"].to_numpy()[0])
                else:
                    st.warning("Incorrect Email/Password")
        else:
            st.warning("Not Valid Email")
                
           
if choice=="SignUp":
    Fname = st.text_input("First Name")
    Lname = st.text_input("Last Name")
    Mname = st.text_input("Mobile Number")
    Email = st.text_input("Email")
    City = st.text_input("City")
    Password = st.text_input("Password",type="password")
    CPassword = st.text_input("Confirm Password",type="password")
    b2=st.button("SignUp")
    if b2:
        pattern=re.compile("(0|91)?[7-9][0-9]{9}")
        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if Password==CPassword:
            if (pattern.match(Mname)):
                if re.fullmatch(regex, Email):
                    create_usertable()
                    add_userdata(Fname,Lname,Mname,City,Email,Password,CPassword)
                    st.success("SignUp Success")
                    st.info("Go to Logic Section for Login")
                else:
                    st.warning("Not Valid Email")         
            else:
                st.warning("Not Valid Mobile Number")
        else:
            st.warning("Pass Does Not Match")
