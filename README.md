# MulticoreCNNProject
*Sejong Univ. Multicore Programming*

OpenCL gpu 병렬 프로그래밍을 활용하여 CNN(Convolutional Neural Network) 알고리즘을 가속화하는 프로젝트입니다.

![image](https://github.com/sladja3929/MulticoreCNNProject/assets/43125863/ddbe4772-ae9e-4f09-9aa0-c5e3537b21ac)

## 목차
+ 팀 소개
+ 기술 스택
+ 구현 기능
+ 실행 결과


## 팀 소개
팀원 구성: 권오민, 조민수, 나원준
+ 권오민: Pooling Layer 구현
+ 조민수: Convolution Layer 구현
+ 나원준: FC Layer 구현


## 기술 스택
+ 사용 언어: C, C++, OpenCL


## 구현 기능
### Convolution Layer
이미지를 분류하는데 필요한 정보를 필터를 통해 뽑아냄
6중 반복문을 2중 반복문으로 병렬화
global size : {d1, d2 * n * n * batch_num}
local size : {d1, 1}

### Pooling Layer
Convolution Filter를 거친 결과에서 해당 영역 내에서 가장 큰 값을 도출
5중 반복문을 2중 반복문으로 병렬화
global size = {batch_num, d * n * n}
각 barrier마다 1개씩 더하는 방법에서 3개씩 더하여 barrier waiting을 줄임

### FC Layer
Pooling Layer의 값으로 이미지 분류
2중 반복문을 모두 병렬화
global size = {input * weight}
local size = N
Reduction 기법 사용

### image batch
작업 수행 메모리 크기가 적어 동시에 여러 개의 이미지를 처리

### buffer swap
buffer 최적화, 각 함수마다 input 메모리가 output 되는 것을 발견하고 input, output 메모리 버퍼를 스왑하여 readbuffer 오버헤드를 줄임

### 기타 실패한 기법들
+ double buffering: kernel을 2개 사용하여 output의 readbuffer와 Layer 작업을 분리하려 했으나 buffer swap으로 이미 readbuffer의 시간이 단축되어 유의미한 결과를 얻지 못함
+ cpu 쓰레드 분리: FC Layer의 Softmax, Findmax 함수를 gpu 작업이 이루어지는 동안 cpu에서 병렬 쓰레드로 처리하려 했으나 유의미한 결과를 얻지 못함


## 실행 결과
+ 실험 환경: RTX 3060, i7 10100
+ 사용 image: 3000장
+ cpu only: 600s
+ gpu 병렬처리: 15.01s
