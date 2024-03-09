# RACE ENGINEER
### Now you can lose your races in ACC or iRacing but with the help of AI!

This app is still in its production stage.

To run the app, create a ".env" file in this here root folder containing your openai api key.
```
OPENAI_API_KEY="meaningless-string-of-letters"
PINECONE_API_KEY="another-meaningless-string-of-letters"
```

Then, cd to the appCL folder and start the app with this command:
```
$ cd appCL
$ chainlit run app.py

# alternatively, you can run with the watch flag to auto-reload the web app as you make changes
$ chainlit run app.py -w
```