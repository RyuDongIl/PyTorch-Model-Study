# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 21:28:48 2022

@author: Dongil Ryu
"""
import time
import math

import spacy
from konlpy.tag import Komoran
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Field, BucketIterator
from model import Encoder, Decoder, Attention

BATCH_SIZE = 128
kmran = Komoran() # 한국어 토큰화
spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)

# 모델 학습(train) 함수
def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train() # 학습 모드
    epoch_loss = 0
    
    # 전체 학습 데이터를 확인하며
    with tqdm(enumerate(iterator), total=len(iterator), desc=f'Epoch {epoch + 1}', unit='batch', ncols=160) as tepoch:
        for i, batch in tepoch:
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            output = model(src, trg)
            # output: [출력 단어 개수, 배치 크기, 출력 차원]
            output_dim = output.shape[-1]

            # 출력 단어의 인덱스 0은 사용하지 않음
            output = output[1:].view(-1, output_dim)
            # output = [(출력 단어의 개수 - 1) * batch size, output dim]
            trg = trg[1:].view(-1)
            # trg = [(타겟 단어의 개수 - 1) * batch size]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)
            loss.backward() # 기울기(gradient) 계산

            # 기울기(gradient) clipping 진행
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # 파라미터 업데이트
            optimizer.step()

            # 전체 손실 값 계산
            epoch_loss += loss.item()

            tepoch.set_postfix(loss=loss.item())
    
    return epoch_loss / len(iterator)

# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0
    
    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 평가할 때 teacher forcing는 사용하지 않음
            output = model(src, trg, 0)
            # output: [출력 단어 개수, 배치 크기, 출력 차원]
            output_dim = output.shape[-1]
            
            # 출력 단어의 인덱스 0은 사용하지 않음
            output = output[1:].view(-1, output_dim)
            # output = [(출력 단어의 개수 - 1) * batch size, output dim]
            trg = trg[1:].view(-1)
            # trg = [(타겟 단어의 개수 - 1) * batch size]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# 한국어 문장을 토큰화 하는 함수
def tokenize_kr(text):
    return kmran.morphs(text)

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def main():
    SRC = Field(tokenize=tokenize_kr, init_token="<sos>", eos_token="<eos>", lower=True)
    TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)
    train_data, valid_data = TabularDataset.splits(
        path='.', train='data/train.csv', test='data/test.csv', format='csv',
        fields=[('src', SRC), ('trg', TRG)], skip_header=True)

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # 일반적인 데이터 로더(data loader)의 iterator와 유사하게 사용 가능
    train_iterator = BucketIterator(train_data, 
                                    batch_size=BATCH_SIZE,
                                    sort_key=lambda x: len(x['trg']),
                                    device=device)
    valid_iterator = BucketIterator(valid_data, 
                                    batch_size=BATCH_SIZE,
                                    sort_key=lambda x: len(x['trg']),
                                    device=device)

    ## 하이퍼 파라미터 설정
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENCODER_EMBED_DIM = 1000
    DECODER_EMBED_DIM = 1000
    HIDDEN_DIM = 1000
    DEEP_DIM = 500
    ENC_DROPOUT_RATIO = 0.5
    DEC_DROPOUT_RATIO = 0.5
    
    # 인코더(encoder)와 디코더(decoder) 객체 선언
    enc = Encoder(INPUT_DIM, ENCODER_EMBED_DIM, HIDDEN_DIM, ENC_DROPOUT_RATIO)
    dec = Decoder(OUTPUT_DIM, DECODER_EMBED_DIM, HIDDEN_DIM, DEEP_DIM, DEC_DROPOUT_RATIO)
    
    # Attention 객체 선언
    model = Attention(enc, dec, device).to(device)
    
    # Adadelta optimizer로 학습 최적화
    optimizer = optim.Adadelta(model.parameters(), rho=0.95)
    
    # 뒷 부분의 패딩(padding)에 대해서는 값 무시
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    
    N_EPOCHS = 20
    CLIP = 1
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time() # 시작 시간 기록
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, epoch)
        valid_loss = evaluate(model, valid_iterator, criterion)
    
        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'attention.pt')
    
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')

    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    main()