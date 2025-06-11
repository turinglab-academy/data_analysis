import pandas as pd
from konlpy.tag import Okt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models as gensim_modelsy
import pyLDAvis
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 여기서는 가상의 뉴스 기사 데이터를 리스트로 사용.
documents = [
    "애플이 새로운 아이폰 모델을 공개하며 인공지능 기능을 대폭 강화했다. 스마트폰 시장의 경쟁이 더욱 치열해질 전망이다.",
    "삼성전자는 폴더블폰 시장의 선두주자로서 신제품을 출시했다. 디스플레이 기술이 혁신을 주도하고 있다.",
    "정부가 부동산 시장 안정화를 위한 새로운 정책을 발표했다. 주택 공급 확대와 투기 억제에 초점을 맞춘다.",
    "한국은행이 기준금리를 인상하여 물가 상승 압력을 억제하려 한다. 경제 전문가들은 가계 부채 증가에 우려를 표했다.",
    "최신 기술 트렌드인 메타버스와 가상현실이 주목받고 있다. 엔터테인먼트 산업에도 큰 영향을 미칠 것으로 예상된다.",
    "환경 오염 문제가 심각해지면서 친환경 에너지 개발의 중요성이 강조되고 있다. 태양광과 풍력 발전이 대안으로 떠오른다.",
    "전기차 판매량이 급증하며 자동차 산업의 패러다임이 변화하고 있다. 배터리 기술 발전이 핵심 경쟁력이다.",
    "국회에서 새로운 법안이 통과되었다. 사회 복지 예산이 증액되어 취약계층 지원이 강화될 예정이다.",
    "인공지능 기술이 의료 분야에 적용되면서 진단 정확도가 향상되고 있다. 개인 맞춤형 치료 시대가 열릴 것이다.",
    "글로벌 경제의 불확실성이 커지면서 환율 변동성이 확대되고 있다. 기업들은 리스크 관리에 집중하고 있다."
 ]
 # --- 3. 형태소 분석 및 전처리 --
# Okt 형태소 분석기 초기화
okt = Okt()
 # 불용어 리스트 (분석 목적에 따라 추가/수정)
stop_words = ['은', '는', '이', '가', '을', '를', '에', '와', '과', '하다', '이다', '되다', '으로', '에서', '에게', '및', '또는', '등', '있다', '없다', '것', '수', '고', '면', 
'며', '아', '어야', '어서', '다', '았', '었', '겠', '습니다', '합니다', '한다', '하는', 
'될', '기', '저', '그', '더', '좀', '가장', '매우', '아주', '특히', '점', '때문']

processed_docs = []
for doc in documents:
    # 명사만 추출
    nouns = okt.nouns(doc)
    # 불용어 제거 및 한 글자 단어 제거
    filtered_nouns = [word for word in nouns if word not in stop_words and len(word) > 
1]
    processed_docs.append(filtered_nouns)

print("--- 전처리된 문서 예시 (첫 3개) ---")
for i, doc in enumerate(processed_docs[:3]):
    print(f"문서 {i+1}: {doc}")
print("\n")
# --- 4. 사전 (Dictionary) 및 코퍼스 (Corpus) 생성 --
# Gensim Dictionary 생성: 각 단어에 고유 ID 부여
dictionary = corpora.Dictionary(processed_docs)
 # DTM (Document-Term Matrix) 또는 BoW (Bag of Words) 코퍼스 생성

# 각 문서의 단어들을 (단어_ID, 출현_빈도) 형태로 변환
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print("--- Corpus 예시 (첫 문서) ---")
print(corpus[0]) # (단어_ID, 빈도) 튜플 리스트
print("--- 단어 ID -> 단어 매핑 예시 ---")
print(dictionary[0], dictionary[1]) # ID 0과 1에 해당하는 단어
print("\n")
 # --- 5. LDA 모델 학습 --
num_topics = 3 # 발견하고자 하는 토픽의 개수 설정 (조절 가능)
passes = 10    
# 학습 반복 횟수 (더 많을수록 모델 안정화에 도움)
iterations = 500 # 각 문서에 대한 토픽 할당 반복 횟수
# LDA 모델 학습
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary,
passes=passes, iterations=iterations, random_state=42)
print("--- LDA 모델 학습 결과 (상위 토픽 3개) ---")

for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}")
    print("\n")

# --- 6. pyLDAvis를 이용한 시각화 --
# LDA 모델 결과를 pyLDAvis 포맷으로 변환
lda_display = gensim_modelsy.prepare(lda_model, corpus, dictionary, sort_topics=False)

# HTML 파일로 저장 (브라우저에서 열어 확인)
pyLDAvis.save_html(lda_display, 'lda_topics_visualization.html')
print("pyLDAvis 시각화 결과가 'lda_topics_visualization.html' 파일로 저장되었습니다.")
print("이 파일을 웹 브라우저로 열어 토픽 분포를 시각적으로 확인하세요.\n")

# --- 7. 각 문서의 토픽 분포 확인 (선택 사항) --
print("--- 각 문서의 토픽 분포 ---")
for i, doc_bow in enumerate(corpus):
    doc_topics = lda_model.get_document_topics(doc_bow)
    # 각 토픽과 해당 토픽이 문서에 기여하는 확률 출력
    topic_distribution = sorted(doc_topics, key=lambda x: x[1], reverse=True)
    # 원본 문서 내용 출력
    print(f"원본 문서 {i+1}: {documents[i][:50]}...") # 긴 문서일 경우 일부만 출력

# 토픽 분포 출력
topic_str = []
for topic_id, prob in topic_distribution:
    topic_str.append(f"Topic {topic_id}: {prob:.3f}")
    print(f"  토픽 분포: {', '.join(topic_str)}")
    print("-" * 30)

