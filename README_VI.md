# Chạy Scenario 3 với JGEA

## 1) Chuẩn bị
- Cài Java JDK 21.
- Cài Maven 3.9.11 hoặc mới hơn.
- Giải nén `jgea-main.zip` thành thư mục `jgea-main`.
- Copy 4 file này vào trong thư mục `jgea-main`:
  - `scenario3_quick_test.txt`
  - `scenario3_paper_like.txt`
  - `run_windows.bat`
  - `run_linux.sh`

## 2) Chạy thử nhanh
Windows:
```bat
run_windows.bat
```
Linux/macOS:
```bash
./run_linux.sh
```

## 3) Chạy full giống paper
Mở file script và bỏ comment dòng chạy `scenario3_paper_like.txt`.
Full run = 9 problems x 9 solvers x 30 seeds = 2430 runs, mỗi run paper báo khoảng 100-200 giây trên Xeon W-2295. Máy i7/32GB chạy được, nhưng có thể mất nhiều thời gian. RTX 3050 hầu như không được dùng vì mô phỏng/JGEA chạy CPU.

## 4) Kết quả
Kết quả và hình `best-traj*.svg` nằm trong thư mục `out/`.
