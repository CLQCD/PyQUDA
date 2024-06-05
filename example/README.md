# 简介与示例

## 1. 纯规范 HMC，包含费米子的 HMC（可选）

```bash
python3 1.py
```

产生的组态文件保存至 `DATA` 文件夹

## 2. 在产生的规范场组态上做HYP smearing，然后算静态势并大致估计格距

## 3. 在HYP smeared的组态上跑clover传播子（csw=1），计算pion和nucleon的2pt（mpi～300 MeV，700MeV，3000MeV）

## 4. 基于PCAC计算三种pion mass下的quark mass

## 5. 计算不同动量的2pt，确定色散关系，计算quasi-DA矩阵元（可选）

## 6. 计算seq source，计算nucleon的gV和gA

## 7. 计算quasi-PDF矩阵元（可选）
