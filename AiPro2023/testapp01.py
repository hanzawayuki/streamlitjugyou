# 必要なライブラリをインポートする
import streamlit as st
import numpy as np
import pandas as pd

# 起動方法
# > streamlit run testapp01.py

# サーバへのアクセス方法
# > Local URL: をブラウザで起動する

# コードの変更時
# > ブラウザの画面をリロードすると、変更が反映されます

# 終了方法
# > (Ctrl) + (c)キー

#=============================================================================#
# 以降のコードは順次、レッスン単位でコメントアウトを解除しながら試行してください
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#

# # Streamlitの基礎:レッスン1
# # 文字を表示する
# st.title('レッスン1')           # タイトル
# st.header('レッスン1')          # ヘッダ
# st.subheader('レッスン1')       # サブヘッダ
# st.caption('レッスン1')         # キャプション
# st.code('print("レッスン1"')    # ソースコード
# st.write('レッスン1')           # 汎用的な出力
# st.latex('S_{t+1}=S_{t}\exp(\mu \Delta_t+\sigma \sqrt{\Delta_t}\epsilon_t)')    # 数式（ラテック形式/tex:テフ）

# # Streamlitの基礎:レッスン2
# # 数値を入力する
# n = st.number_input(label='数値を入力してください', value=7)
# st.write(n)

# # Streamlitの基礎:レッスン3
# # 簡単な計算を実行する
# n1 = st.number_input(label='数値1を入力してください', value=0)
# n2 = st.number_input(label='数値2を入力してください', value=7)
# n3 = n1 + n2
# mess = '数値1＋数値2＝' + str(n3)
# st.write(mess)


#=============================================================================#
# 課題. Streamlitの標準UIを試そう
# 初級（必須）：1～2種類のUIを実装した
# 中級（任意）：3～5種類のUIを実装した
# 上級（任意）：6種類以上のUIを実装した
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#
# Streamlitの公式ドキュメント「API reference」↓
# https://docs.streamlit.io/library/api-reference
# から、好きなUIをクリックする → コードの記述例(Example)をコピーする → 動作確認する
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#


### ここから(課題.冒頭)　###

#ダウンロード
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(my_large_df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)

#セレクトボタン
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col1, col2 = st.columns(2)

with col1:
    st.checkbox("Disable selectbox widget", key="disabled")
    st.radio(
        "あなたのよく使う検索エンジン👉",
        key="visibility",
        options=["edge", "chrome", "google"],
    )

with col2:
    option = st.selectbox(
        "あなたが今使っている媒体",
        ("PC", "ノートPC", "スマホ"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

#ラジオボタン
genre = st.radio(
    "あなたの好きな映画のジャンル",
    ('コメディ', 'ドラマ', 'ドキュメンタリー'))

if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write("You didn\'t select comedy.")

#誕生日とカレンダー
import streamlit as st

d = st.date_input(
    "When\'s your birthday",
    datetime.date(2002, 1, 610))
st.write('Your birthday is:', d)

#色変え
color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

#テキストエリア
txt = st.text_area('Text to analyze', '''
    It was the best of times, it was the worst of times, it was
    the age of wisdom, it was the age of foolishness, it was
    the epoch of belief, it was the epoch of incredulity, it
    was the season of Light, it was the season of Darkness, it
    was the spring of hope, it was the winter of despair, (...)
    ''')
st.write('Sentiment:', run_sentiment_analysis(txt))



if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('goodbye')

### ここまで(課題.末尾) ###


#=============================================================================#
# 提出方法：以下の項目を記入の上、Teamsの課題機能で、このコードファイルを提出する
# 氏名：半澤優希
# 学科：高度情報工学科
# 学年：4年
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#

