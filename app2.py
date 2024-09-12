import plotly.express as px
import plotly
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# 한글 폰트 설정 : 한글 깨짐을 방지하기 위함
plt.rc('font', family='Malgun Gothic')

money = pd.read_csv("c:\\data\\money_data6.csv")

plt.figure(figsize = (12, 8))

fig, ax = plt.subplots()
ax.plot( list(money['A_MONTH']), list(money['A_RATE']), color = 'red', marker = 'o' )
plt.xticks(tuple(money['A_MONTH']))
plt.title("America Rate", size = 15 )

st.pyplot(fig)
