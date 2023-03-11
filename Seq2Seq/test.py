# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:26:50 2022

@author: Dongil Ryu
"""
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Field, BucketIterator
from model import Encoder, Decoder, Seq2Seq

BATCH_SIZE = 1
spacy_de = spacy.load('de_core_news_sm') # 독일어 토큰화(tokenization)
spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)

# 독일어(Deutsch) 문장을 토큰화 하는 함수
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)][::-1]


# 번역(translation) 함수
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval() # 평가 모드

    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_en.tokenizer(sentence)][::-1]
    else:
        tokens = [token.lower() for token in sentence]

    # 처음에 <sos> 토큰, 마지막에 <eos> 토큰 붙이기
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    print(f"전체 소스 토큰: {tokens}")

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    print(f"소스 문장 인덱스: {src_indexes}")

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    # 인코더(endocer)에 소스 문장을 넣어 문맥 벡터(context vector) 계산
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    # 처음에는 <sos> 토큰 하나만 가지고 있도록 하기
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        # 이전에 출력한 단어가 현재 단어로 입력될 수 있도록
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # <eos>를 만나는 순간 끝
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # 각 출력 단어 인덱스를 실제 단어로 변환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    return trg_tokens[1:]

def test():
    SRC = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)
    TRG = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True)
    train_data, valid_data = TabularDataset.splits(
        path='.', train='data/train.csv', test='data/test.csv', format='csv',
        fields=[('src', SRC), ('trg', TRG)], skip_header=True)

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    
    ## 하이퍼 파라미터 설정
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENCODER_EMBED_DIM = 256
    DECODER_EMBED_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT_RATIO = 0.5
    DEC_DROPOUT_RATIO = 0.5
    
    
    # 인코더(encoder)와 디코더(decoder) 객체 선언
    enc = Encoder(INPUT_DIM, ENCODER_EMBED_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT_RATIO)
    dec = Decoder(OUTPUT_DIM, DECODER_EMBED_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT_RATIO)
    
    # Seq2Seq 객체 선언
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load('seq2seq_real.pt', map_location=device))
    
    idx = 24
    src = valid_data[idx].src
    trg = valid_data[idx].trg
    
    print(f'소스 문장: {src}')
    print(f'타겟 문장: {trg}')
    print("모델 출력 결과:", " ".join(translate_sentence(src, SRC, TRG, model, device)))
    
if __name__ == '__main__':
    test()
    