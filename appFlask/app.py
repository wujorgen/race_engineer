from flask import Flask, render_template

# import streamlit as st
import chainlit

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/streamlit")
# def streamlit():
    # st.set_page_config(page_title="My Streamlit App")
    # st.write("Hello, world!")


if __name__ == "__main__":
    app.run(debug=True)
