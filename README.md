# Distributed LLM Training and Sliding Window Data Preparation

Author - Gautham Satyanarayana <br />
Email - gsaty@uic.edu <br />
UIN - 659368048

## Introduction
As part of UIC-CS441 Engineering Distributed Objects for Cloud Computing, 
this project demonstrates how to train an LLM. 
For the second Homework, we build a spark job to tokenise, generate sliding window examples over an input text corpus 
and finally train a distributed LLM and monitor training metrics. 
<br /><br />
<b>Video Link</b>: 

## Environment
- MacOSX ARM64
- IntelliJ IDEA 2024.2.1
- Spark 3.5.2
- Scala v2.13.12
- SBT v1.10.2

## Training Parameters and Frameworks
- Dataset - Books from Project Gutenberg
- Preprocessing Strategy - Remove punctuation, numbers and force to lowercase, tokenise
- Training - Dl4j-spark
- Testing Framework - munit
- Config - TypeSafe Config
- Logging - Sl4j


## Data Flow and Logic

## EMR Flow
Explained in the YouTube video linked above

## Test Suite
Tests are found in `/src/test/scala/Suite.scala` or just run from root

```angular2html
sbt test
```

## Results

## Usage

