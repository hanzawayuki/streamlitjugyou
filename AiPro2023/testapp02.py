# 必要なライブラリをインポートする
import streamlit as st
import numpy as np
import pandas as pd

# 起動方法
# > streamlit run testapp02.py

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
# 課題. 簡単な計算機を作成しよう
# 初級（必須）：2つの数の足し算、引き算、掛け算、割り算をする
# 中級（任意）：四則演算のうち、どから1つを選択して実行できる
# 上級（任意）：自由課題 ※累乗や平方根など、オリジナルの機能を創意工夫してみよう
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#
# 上の方の級が解けた場合は、下の方の級のコードは残しておかなくてもOKです
# （中級が解ける人は初級は解けるし、上級が解ける人は、初・中級は解けるでしょうから）
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#
# 中級を解く際に、参考にしてほしいページをご紹介（公式ドキュメント）
# https://docs.streamlit.io/library/api-reference/widgets/st.radio
#
# 上級を解く際に、参考になるかもしれないサイトをご紹介
# https://blog.amedama.jp/entry/streamlit-tutorial
# https://data-analytics.fun/2022/01/29/understanding-streamlit-1/
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#


### ここから(課題1.冒頭)　###
n1 = st.number_input(label='足し算です数値1を入力してください', value=0)
n2 = st.number_input(label='足し算です数値2を入力してください', value=7)
n3 = n1 + n2
mess = '数値1＋数値2＝' + str(n3)
st.write(mess)
n4 = st.number_input(label='引き算です数値1を入力してください', value=0)
n5 = st.number_input(label='引き算です数値2を入力してください', value=7)
n6 = n4 - n5
mess2 = '数値1ー数値2＝' + str(n6)
st.write(mess2)
n7 = st.number_input(label='掛け算です数値1を入力してください', value=0)
n8 = st.number_input(label='掛け算です数値2を入力してください', value=7)
n9 = n7 * n8
mess3 = '数値1×数値2＝' + str(n9)
st.write(mess3)
n9 = st.number_input(label='割り算です数値1を入力してください', value=0)
n10 = st.number_input(label='割り算です数値2を入力してください', value=7)
n11 = n9 / n10
mess4 = '数値1÷数値2＝' + str(n11)
st.write(mess4)
### ここまで(課題1.末尾) ###


#=============================================================================#
# 提出方法：以下の項目を記入の上、Teamsの課題機能で、このコードファイルを提出する
# 氏名：半澤優希
# 学科：高度情報工学科
# 学年：４年
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#

