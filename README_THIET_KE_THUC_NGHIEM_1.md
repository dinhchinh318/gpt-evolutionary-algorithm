# README — Thiết kế thực nghiệm 1: Synthetic Problems bằng JGEA trên thiết bị cá nhân

Tài liệu này mô tả đầy đủ **thiết kế thực nghiệm 1** trong project `jgea.zip` bạn gửi: thực nghiệm tối ưu các hàm synthetic bằng các Evolutionary Algorithms (EA) trong JGEA. Nội dung được viết theo đúng các file hiện có trong project, đặc biệt là:

- `scenario1.txt`: cấu hình thực nghiệm chính.
- `run_windows_scenario1.bat`: script chạy trên Windows.
- `out/Scenario_1.csv`: log theo từng iteration.
- `out/Scenario_1_final.csv`: kết quả cuối của từng run.
- Source code JGEA trong `io.github.ericmedvet.jgea.problem` và `io.github.ericmedvet.jgea.experimenter`.
- Ảnh cấu hình máy bạn cung cấp.

---

## 1. Mục tiêu của thực nghiệm 1

Thực nghiệm 1 kiểm tra khả năng của nhiều thuật toán tiến hóa khác nhau khi giải các bài toán tối ưu số học nhân tạo.

Bài toán tổng quát là:

```text
Tìm vector x sao cho f(x) nhỏ nhất.
```

Viết dưới dạng toán học:

```text
minimize f(x), với x ∈ R^p
```

Trong đó:

- `x` là một nghiệm ứng viên.
- `p` là số chiều của nghiệm, tức số biến cần tối ưu.
- `f(x)` là hàm mục tiêu.
- Giá trị `f(x)` được gọi là fitness hoặc quality.
- Vì đây là bài toán minimization, `f(x)` càng nhỏ càng tốt.

Thực nghiệm này không dùng neural network, không dùng dataset, không dùng robot. Nó chỉ dùng các hàm toán học synthetic để kiểm tra năng lực search của EA.

Ý nghĩa chính:

- Kiểm tra EA có tìm được nghiệm tốt không.
- So sánh EA nào hội tụ tốt hơn.
- Xem EA hoạt động thế nào khi số chiều tăng từ `p = 20` lên `p = 500`.
- Kiểm tra trên nhiều loại landscape: dễ, khó, unimodal, multimodal, có nhiều local optima, nghiệm gần gốc, nghiệm xa gốc.

---

## 2. Thiết bị chạy thực nghiệm

Theo ảnh cấu hình máy bạn cung cấp:

| Thành phần | Thông tin |
|---|---|
| Laptop | Acer Nitro AN515-58 |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700H, 2.70 GHz |
| RAM | 32 GB, khoảng 31.7 GB usable |
| GPU rời | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB |
| GPU tích hợp | Intel(R) Iris(R) Xe Graphics |
| Storage | 477 GB tổng, khoảng 425 GB đã dùng tại thời điểm chụp |
| System type | Windows 64-bit, x64-based processor |

### Ý nghĩa đối với thực nghiệm

Thực nghiệm JGEA này chủ yếu chạy trên **CPU**, không phải GPU. Các EA trong Scenario 1 chỉ tính toán trên vector số thực và các hàm toán học như Sphere, Ackley, Rastrigin, Griewank. Do đó:

- RTX 3050 hầu như không giúp tăng tốc thực nghiệm này.
- CPU i7-12700H là phần quan trọng nhất.
- RAM 32 GB đủ để chạy cấu hình `-Xmx24g` trong file `.bat`.
- Nên đóng bớt ứng dụng nặng khi chạy vì script cho Java heap tối đa 24 GB.
- Máy có thể nóng và quạt chạy mạnh vì toàn bộ workload là CPU-bound.

---

## 3. File quan trọng trong project

Trong `jgea.zip`, các file liên quan trực tiếp đến thực nghiệm 1 là:

```text
jgea/
├── scenario1.txt
├── run_windows_scenario1.bat
├── out/
│   ├── Scenario_1.csv
│   └── Scenario_1_final.csv
├── io.github.ericmedvet.jgea.experimenter/
│   └── target/
│       └── jgea.experimenter-2.8.1-jar-with-dependencies.jar
└── io.github.ericmedvet.jgea.problem/
    └── src/main/java/io/github/ericmedvet/jgea/problem/synthetic/numerical/
        ├── Sphere.java
        ├── PointsAiming.java
        ├── CircularPointsAiming.java
        ├── Ackley.java
        ├── Rastrigin.java
        ├── Rosenbrock.java
        └── Griewank.java
```

Ý nghĩa:

| File | Vai trò |
|---|---|
| `scenario1.txt` | Định nghĩa toàn bộ thiết kế thực nghiệm 1: problems, solvers, seeds, listeners, output CSV |
| `run_windows_scenario1.bat` | Script chạy Scenario 1 trên Windows |
| `Scenario_1.csv` | Log theo thời gian/iteration, ghi nhiều dòng trong quá trình EA chạy |
| `Scenario_1_final.csv` | Chỉ lấy dòng cuối của mỗi run, dùng để so sánh kết quả cuối |
| `jgea.experimenter-2.8.1-jar-with-dependencies.jar` | File JAR dùng để chạy experiment |
| Source `synthetic/numerical` | Nơi định nghĩa công thức các hàm benchmark |

---

## 4. Input của thực nghiệm là gì?

Một run của thực nghiệm nhận các input sau:

```text
1. Problem: một hàm synthetic cụ thể
2. Dimension p: số chiều của vector nghiệm
3. Solver: một thuật toán EA cụ thể
4. Seed: seed ngẫu nhiên để lặp lại thí nghiệm
5. Stop condition: tối đa 10,000 fitness evaluations
6. Initial distribution: nghiệm ban đầu sinh trong [-1, 1]^p
```

Ví dụ một run cụ thể:

```text
problem = Rastrigin-100
solver = ga-0.02
seed = 7
p = 100
stop = 10,000 evaluations
```

EA sẽ tìm một vector:

```text
x = [x1, x2, ..., x100]
```

sao cho:

```text
Rastrigin(x) nhỏ nhất có thể.
```

---

## 5. Nghiệm ứng viên là gì?

Trong Scenario 1, mỗi nghiệm ứng viên là một vector số thực có độ dài `p`:

```text
x ∈ R^p
```

Ví dụ với `p = 20`:

```text
x = [x1, x2, x3, ..., x20]
```

Với `p = 500`:

```text
x = [x1, x2, x3, ..., x500]
```

Trong JGEA, nghiệm này được biểu diễn là `List<Double>`.

Hàm `AbstractNumericalProblem` kiểm tra độ dài vector đầu vào. Nếu vector không có đúng `p` phần tử thì lỗi.

---

## 6. EA có tự sinh nghiệm không?

Có. EA tự sinh ra các bộ nghiệm ban đầu.

Trong `scenario1.txt`, phần factory được định nghĩa là:

```text
$factory = ea.r.f.dsUniform(initialMinV = -1.0; initialMaxV = 1.0)
```

Nghĩa là các phần tử của vector nghiệm ban đầu được sinh ngẫu nhiên trong khoảng:

```text
[-1, 1]
```

Ví dụ với `p = 5`, một population ban đầu có thể gồm:

```text
x1 = [ 0.25, -0.71,  0.10,  0.93, -0.42]
x2 = [-0.18,  0.44, -0.90,  0.11,  0.67]
x3 = [ 0.76,  0.21,  0.33, -0.55, -0.02]
...
```

EA không biết nghiệm tối ưu nằm ở đâu. Nó chỉ biết đánh giá từng vector bằng hàm mục tiêu `f(x)`.

---

## 7. EA làm gì từ đầu đến cuối?

Quá trình chung của một EA trong Scenario 1:

```text
Bước 1: Sinh population ban đầu
Bước 2: Tính fitness f(x) cho từng nghiệm
Bước 3: Xác định nghiệm tốt hơn dựa trên fitness nhỏ hơn
Bước 4: Tạo nghiệm mới bằng mutation, crossover, update rule hoặc sampling
Bước 5: Tính fitness cho nghiệm mới
Bước 6: Cập nhật population
Bước 7: Lặp lại đến khi đạt 10,000 evaluations
Bước 8: Trả về nghiệm tốt nhất tìm được và fitness tương ứng
```

Sơ đồ:

```text
Input problem f(x), p, EA
        |
        v
Generate initial population x ∈ R^p
        |
        v
Evaluate fitness f(x)
        |
        v
Select / update / mutate / crossover
        |
        v
Generate new candidate solutions
        |
        v
Evaluate again
        |
        v
Repeat until 10,000 evaluations
        |
        v
Output best_x and best_fitness
```

---

## 8. Output của một run là gì?

Output trực tiếp của một run là:

```text
best_x
best_fitness = f(best_x)
```

Ví dụ:

```text
Problem: Rastrigin-100
Solver: ga-0.02
Seed: 7

best_x = [0.02, -0.01, 0.00, ..., 0.03]
best_fitness = 5.31
```

Nếu global optimum lý thuyết là:

```text
x* = [0, 0, ..., 0]
f(x*) = 0
```

thì `best_fitness = 5.31` nghĩa là EA đã tìm được nghiệm khá gần optimum, nhưng chưa hoàn hảo.

---

## 9. Output của toàn bộ thực nghiệm là gì?

Toàn bộ Scenario 1 xuất ra hai file CSV chính:

### 9.1. `out/Scenario_1.csv`

File này ghi log trong quá trình chạy, không chỉ dòng cuối.

Trong project hiện tại, file này có khoảng:

```text
3,041,520 dòng dữ liệu
```

Nó dùng để vẽ các đường hội tụ như:

```text
best fitness theo số evaluations
```

### 9.2. `out/Scenario_1_final.csv`

File này chỉ giữ dòng cuối của mỗi run.

Trong project hiện tại, file này có:

```text
8,640 dòng
```

Con số này đến từ:

```text
30 seeds × 32 problems × 9 solvers = 8,640 runs
```

File này phù hợp để vẽ:

- boxplot final best fitness;
- bảng so sánh solver;
- ranking solver;
- heatmap solver × problem;
- thống kê median/mean/std.

---

## 10. Vì sao là 32 problems?

Trong paper gốc, Scenario 1 thường gồm:

```text
Sphere, PA-1, PA-10, CPA, Ackley, Rastrigin
```

với 4 giá trị `p`, tức:

```text
6 nhóm × 4 p = 24 problems
```

Nhưng trong file `scenario1.txt` của project bạn, bạn đã mở rộng thêm:

```text
Rosenbrock
Griewank
```

Do đó project hiện tại có:

```text
8 nhóm function × 4 p = 32 problems
```

Danh sách đầy đủ:

| Nhóm hàm | p = 20 | p = 100 | p = 200 | p = 500 |
|---|---:|---:|---:|---:|
| Sphere | ✓ | ✓ | ✓ | ✓ |
| PA-1 | ✓ | ✓ | ✓ | ✓ |
| PA-10 | ✓ | ✓ | ✓ | ✓ |
| CPA | ✓ | ✓ | ✓ | ✓ |
| Ackley | ✓ | ✓ | ✓ | ✓ |
| Rastrigin | ✓ | ✓ | ✓ | ✓ |
| Rosenbrock | ✓ | ✓ | ✓ | ✓ |
| Griewank | ✓ | ✓ | ✓ | ✓ |

---

## 11. Các hàm benchmark trong thực nghiệm

### 11.1. Sphere

Công thức trong code:

```text
f(x) = sum(x_i^2)
```

Tính chất:

- Unimodal.
- Một global optimum tại `x = [0, 0, ..., 0]`.
- `f(x*) = 0`.
- Đây là bài toán dễ nhất.

Ý nghĩa:

```text
Kiểm tra EA có khả năng hội tụ cơ bản không.
```

---

### 11.2. Point Aiming PA-1

Cấu hình:

```text
ea.p.s.pointAiming(name = "PA-1-p"; p = p; target = 1.0)
```

Công thức:

```text
f(x) = ||x - target||_2
```

với:

```text
target = [1, 1, ..., 1]
```

Tính chất:

- Unimodal.
- Một global optimum tại `[1, 1, ..., 1]`.
- Nghiệm tối ưu nằm gần vùng khởi tạo `[-1, 1]^p`.

Ý nghĩa:

```text
Kiểm tra EA tìm target gần gốc tọa độ tốt không.
```

---

### 11.3. Point Aiming PA-10

Cấu hình:

```text
ea.p.s.pointAiming(name = "PA-10-p"; p = p; target = 10.0)
```

Target:

```text
[10, 10, ..., 10]
```

Tính chất:

- Unimodal.
- Một global optimum tại `[10, 10, ..., 10]`.
- Khó hơn PA-1 vì nghiệm nằm xa vùng khởi tạo `[-1, 1]^p`.

Ý nghĩa:

```text
Kiểm tra EA có di chuyển đủ xa khỏi vùng khởi tạo để tìm optimum không.
```

---

### 11.4. CPA — Circular Points Aiming

Cấu hình:

```text
ea.p.s.circularPointsAiming(
  name = "CPA-p";
  p = p;
  n = 5;
  radius = 2.0;
  center = 1.0;
  seed = 1
)
```

Trong project này, CPA không phải là “cả vòng tròn đều tối ưu”. Code `CircularPointsAiming.java` tạo **5 target points** bằng cách:

1. Sinh 5 vector hướng ngẫu nhiên trong không gian p chiều.
2. Chuẩn hóa mỗi vector thành unit vector.
3. Nhân với `radius = 2.0`.
4. Dịch tâm bằng `center = 1.0`.

Hàm mục tiêu kế thừa từ `PointsAiming`:

```text
f(x) = min distance từ x đến một trong các target points
```

Tính chất:

- Có 5 global optima.
- Multimodal theo nghĩa có nhiều target tối ưu.
- Không nên diễn giải là vô số điểm trên đường tròn, vì code hiện tại dùng 5 điểm target rời rạc.

Ý nghĩa:

```text
Kiểm tra EA có tìm được một trong nhiều target tối ưu không.
```

---

### 11.5. Ackley

Công thức trong code:

```text
f(x) = -20 exp(-0.2 sqrt(sum(x_i^2) / p))
       - exp(sum(cos(2πx_i)) / p)
       + 20 + e
```

Tính chất:

- Multimodal.
- Có nhiều local optima.
- Global optimum tại `[0, 0, ..., 0]`.
- `f(x*) = 0`.

Ý nghĩa:

```text
Kiểm tra EA có bị mắc kẹt ở local optima không.
```

---

### 11.6. Rastrigin

Công thức trong code:

```text
f(x) = 10p + sum(x_i^2 - 10 cos(2πx_i))
```

Tính chất:

- Multimodal.
- Có rất nhiều local optima.
- Global optimum tại `[0, 0, ..., 0]`.
- `f(x*) = 0`.
- Landscape gồ ghề, lặp theo chu kỳ.

Ý nghĩa:

```text
Kiểm tra khả năng global search của EA trên landscape nhiều bẫy.
```

---

### 11.7. Rosenbrock

Công thức trong code:

```text
f(x) = sum_{i=1}^{p-1} [100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2]
```

Tính chất:

- Thường được xem là unimodal trong benchmark cơ bản.
- Global optimum tại `[1, 1, ..., 1]`.
- `f(x*) = 0`.
- Khó vì có thung lũng cong hẹp.

Ý nghĩa:

```text
Kiểm tra EA có đi theo được một valley hẹp và cong để tới optimum không.
```

---

### 11.8. Griewank

Công thức trong code:

```text
f(x) = sum(x_i^2) / 4000 - prod(cos(x_i / sqrt(i+1))) + 1
```

Tính chất:

- Multimodal.
- Non-separable.
- Có nhiều local optima do tích cosine.
- Global optimum tại `[0, 0, ..., 0]`.
- `f(x*) = 0`.

Ý nghĩa:

```text
Kiểm tra EA trên landscape dao động nhưng thường mượt hơn Rastrigin.
```

---

## 12. Các EA được so sánh

Trong `scenario1.txt`, có 9 solver:

| Solver | Tên trong CSV | Tham số chính | Ý nghĩa |
|---|---|---|---|
| CMA-ES | `cmaEs` | default | Tối ưu phân phối, tự điều chỉnh covariance |
| Differential Evolution | `de` | NP=15, F=0.5, CR=0.8 | Tạo nghiệm mới bằng khác biệt vector và crossover |
| PSO | `pso` | nPop=100, w=0.8, phi=1.5 | Swarm/particles bay theo best cá nhân và global best |
| Simple ES | `simpleEs-0.02` | sigma=0.02 | ES bước nhỏ |
| Simple ES | `simpleEs-0.25` | sigma=0.25 | ES bước vừa |
| Simple ES | `simpleEs-0.5` | sigma=0.5 | ES bước lớn |
| GA | `ga-0.02` | mutation sigma=0.02, crossoverP=0.8, nPop=100 | GA bước nhỏ |
| GA | `ga-0.25` | mutation sigma=0.25, crossoverP=0.8, nPop=100 | GA bước vừa |
| GA | `ga-0.5` | mutation sigma=0.5, crossoverP=0.8, nPop=100 | GA bước lớn |

---

## 13. Stop condition

Trong `scenario1.txt`:

```text
$stop = ea.sc.nOfQualityEvaluations(n = 10000)
```

Nghĩa là mỗi run dừng khi số lần đánh giá fitness đạt khoảng 10,000.

Trong file kết quả thực tế, cột `n.evals` nằm trong khoảng:

```text
10000 đến 10013
```

Việc vượt nhẹ 10,000 là bình thường vì một số solver sinh/evaluate theo batch hoặc population.

---

## 14. Số lượng run

Cấu hình run:

```text
seed = [1:1:30]
problem = 32 problems
solver = 9 solvers
```

Tổng số run:

```text
30 × 32 × 9 = 8640 runs
```

Trong đó:

- 30 seed giúp đo độ ổn định do EA có tính ngẫu nhiên.
- 32 problems bao phủ nhiều dạng function và nhiều số chiều.
- 9 solvers giúp so sánh nhiều EA.

---

## 15. Một run cụ thể hoạt động như thế nào?

Ví dụ:

```text
problem = Rastrigin-100
solver = ga-0.25
seed = 10
```

Quy trình:

```text
1. JGEA tạo problem Rastrigin với p = 100.
2. GA tạo population ban đầu gồm 100 vector, mỗi vector dài 100.
3. Mỗi phần tử trong vector được sinh trong [-1, 1].
4. Với mỗi vector x, tính Rastrigin(x).
5. Fitness càng nhỏ thì nghiệm càng tốt.
6. GA chọn parent bằng tournament selection.
7. GA tạo offspring bằng mutation hoặc crossover.
8. Offspring được evaluate bằng Rastrigin(x).
9. Parents và offspring được so sánh.
10. Population thế hệ sau giữ các nghiệm tốt hơn.
11. Lặp đến khoảng 10,000 evaluations.
12. Ghi best genotype size và best quality ra CSV.
```

Output cuối cùng là:

```text
best genotype size = 100
best quality = giá trị Rastrigin nhỏ nhất tìm được trong run đó
```

---

## 16. Các cột trong file CSV

Các file CSV có dấu phân tách `;`.

Các cột quan trọng:

| Cột | Ý nghĩa |
|---|---|
| `seed` | seed ngẫu nhiên của run |
| `problem` | tên problem, ví dụ `Rastrigin-100` |
| `solver_sigma` | tên solver, ví dụ `ga-0.25` |
| `n.iterations` | số iteration đã chạy |
| `n.evals` | số lần tính fitness |
| `n.births` | số cá thể/nghiệm đã sinh ra |
| `elapsed.secs` | thời gian chạy run tính bằng giây |
| `best→genotype→size` | số chiều vector nghiệm, chính là p |
| `best→quality` | fitness tốt nhất tìm được |

Cột quan trọng nhất để phân tích kết quả là:

```text
best→quality
```

Vì đây là minimization:

```text
best→quality càng nhỏ càng tốt.
```

---

## 17. Kết quả thực tế trên project hiện tại

Dựa trên `out/Scenario_1_final.csv`:

```text
Số dòng final: 8640
Số seed: 30
Số problem: 32
Số solver: 9
Số dimension: 20, 100, 200, 500
n.evals: 10000 đến 10013
```

Thời gian ghi nhận trong CSV:

```text
median elapsed time / run: khoảng 0.259 giây
min: khoảng 0.059 giây
max: khoảng 95.268 giây
```

Median elapsed time theo solver:

| Solver | Median elapsed seconds |
|---|---:|
| de | 0.2050 |
| pso | 0.2220 |
| simpleEs-0.5 | 0.2740 |
| simpleEs-0.02 | 0.2780 |
| simpleEs-0.25 | 0.2830 |
| ga-0.02 | 0.2895 |
| ga-0.25 | 0.2960 |
| ga-0.5 | 0.3015 |
| cmaEs | 2.3950 |

Lưu ý: thời gian này là số ghi trong CSV của run. Khi chạy toàn bộ experiment trên laptop thực tế, tổng thời gian wall-clock còn phụ thuộc vào Java, số luồng, nhiệt độ máy, tác vụ chạy nền, antivirus, và việc JGEA/executor có chạy song song hay không.

---

## 18. Cách chạy trên máy Windows của bạn

### 18.1. Chạy bằng batch file

Mở terminal trong thư mục `jgea`, sau đó chạy:

```bat
run_windows_scenario1.bat
```

File `.bat` hiện tại chạy lệnh:

```bat
java -Xmx24g -jar "io.github.ericmedvet.jgea.experimenter\target\jgea.experimenter-2.8.1-jar-with-dependencies.jar" -f "scenario1.txt"
```

### 18.2. Chạy trực tiếp bằng command

```bat
java -Xmx24g -jar io.github.ericmedvet.jgea.experimenter\target\jgea.experimenter-2.8.1-jar-with-dependencies.jar -f scenario1.txt
```

### 18.3. Yêu cầu môi trường

Project dùng JGEA `2.8.1`, trong `pom.xml` đặt:

```text
jdk.version = 21
maven.version = 3.9.11
```

Nếu chỉ chạy JAR có sẵn, bạn cần Java runtime phù hợp. Nếu build lại từ source, nên dùng:

```text
JDK 21
Maven 3.9.11 hoặc mới tương thích
```

---

## 19. Lưu ý RAM trên thiết bị của bạn

Máy có 32 GB RAM, script cấp Java heap tối đa:

```text
-Xmx24g
```

Điều này hợp lý nhưng khá cao. Nó để lại khoảng 8 GB cho Windows và ứng dụng khác.

Khuyến nghị:

- Đóng Chrome, IDE nặng, game launcher trước khi chạy.
- Cắm sạc khi chạy.
- Dùng chế độ hiệu năng cao nếu cần.
- Nếu máy lag hoặc thiếu RAM, đổi `-Xmx24g` thành `-Xmx16g`.
- Không nên chạy đồng thời visualization Python nặng khi experiment đang chạy.

---

## 20. Thiết kế thực nghiệm tổng quát

Toàn bộ thiết kế có thể tóm tắt:

```text
For each seed in 1..30:
  For each problem in 32 synthetic problems:
    For each solver in 9 EA solvers:
      1. Initialize population in [-1, 1]^p
      2. Evaluate f(x)
      3. Evolve until 10,000 evaluations
      4. Save iterative log to Scenario_1.csv
      5. Save final best result to Scenario_1_final.csv
```

Tổng số run:

```text
8640
```

Tổng số dòng log iterative:

```text
3,041,520
```

---

## 21. Cách phân tích kết quả

### 21.1. Phân tích final fitness

Dùng `Scenario_1_final.csv`.

Câu hỏi trả lời được:

```text
EA nào tìm được nghiệm cuối tốt hơn?
```

Cách vẽ:

- boxplot `best→quality` theo solver;
- mỗi problem vẽ một subplot;
- giá trị càng thấp càng tốt.

### 21.2. Phân tích convergence

Dùng `Scenario_1.csv`.

Câu hỏi trả lời được:

```text
EA nào hội tụ nhanh hơn?
```

Cách vẽ:

```text
x-axis = n.evals
y-axis = best→quality
line = solver
```

### 21.3. Phân tích ảnh hưởng của số chiều p

Dùng `Scenario_1_final.csv`.

Câu hỏi trả lời được:

```text
Khi p tăng từ 20 lên 500, solver nào bị giảm hiệu quả nhiều hơn?
```

Cách vẽ:

```text
x-axis = p
y-axis = median best→quality
line = solver
facet = function group
```

### 21.4. Phân tích stability

Dùng 30 seeds.

Câu hỏi trả lời được:

```text
EA nào ổn định hơn qua nhiều lần chạy?
```

Cách vẽ:

- boxplot;
- violin plot;
- confidence interval;
- standard deviation theo solver.

---

## 22. Trực quan hóa nên tạo cho báo cáo/slide

Nên tạo các nhóm hình sau.

### 22.1. Sơ đồ Input → EA → Output

Mục đích:

```text
Giải thích EA nhận gì, xử lý gì, trả ra gì.
```

Nội dung:

```text
Input: f(x), p, solver, seed
EA: initialize population, evaluate, select, mutate/crossover/update
Output: best_x, best_fitness, CSV logs
```

### 22.2. Landscape 2D

Mục đích:

```text
Cho thấy hàm dễ/khó thế nào.
```

Cách vẽ:

```text
x-axis = x1
y-axis = x2
color = f(x1, x2, fixed remaining dimensions)
```

Các hình nên có:

- Sphere: một bát trơn.
- PA-1: bát lệch về target `[1,1]`.
- PA-10: target xa vùng khởi tạo.
- CPA: 5 target points.
- Ackley: landscape nhiều dao động.
- Rastrigin: rất nhiều local optima.
- Rosenbrock: valley cong hẹp.
- Griewank: dao động mượt hơn Rastrigin.

### 22.3. Population movement

Mục đích:

```text
Cho thấy EA tự sinh nghiệm và population di chuyển qua generation.
```

Cách vẽ:

- Nền là contour của Rastrigin hoặc Ackley.
- Chấm là các nghiệm trong population.
- Vẽ các snapshot: generation 0, 10, 25, final.
- Đánh dấu global optimum bằng dấu X.

Đây là hình dễ hiểu nhất để giải thích “EA làm gì”.

### 22.4. Convergence curve

Mục đích:

```text
Cho thấy nghiệm tốt nhất cải thiện theo thời gian.
```

Cách vẽ:

```text
x-axis = n.evals
y-axis = best→quality
line = solver
```

### 22.5. Boxplot final fitness

Mục đích:

```text
So sánh kết quả cuối của các EA qua 30 seeds.
```

Cách vẽ:

```text
x-axis = solver
y-axis = best→quality
facet = problem
```

---

## 23. Diễn giải unimodal và multimodal trong thực nghiệm này

### Unimodal

Unimodal nghĩa là landscape có một vùng tối ưu chính. Thuật toán ít bị bẫy local optima.

Trong project:

```text
Sphere
PA-1
PA-10
Rosenbrock, thường xem là unimodal nhưng khó vì valley cong hẹp
```

### Multimodal

Multimodal nghĩa là landscape có nhiều mode, thường là nhiều local optima hoặc nhiều target tối ưu.

Trong project:

```text
CPA
Ackley
Rastrigin
Griewank
```

Lưu ý:

```text
Multimodal không nhất thiết có nhiều global optima.
Ackley, Rastrigin, Griewank thường có 1 global optimum nhưng nhiều local optima.
CPA trong project có 5 global optima vì có 5 target points.
```

---

## 24. Ý nghĩa của từng nhóm solver

### CMA-ES

CMA-ES học phân phối tìm kiếm trong không gian nghiệm. Nó thường mạnh trên nhiều bài toán tối ưu liên tục, nhưng chi phí tính toán có thể cao hơn, đặc biệt khi số chiều lớn.

### DE

DE tạo nghiệm mới bằng cách lấy khác biệt giữa các vector trong population rồi crossover. DE thường mạnh trên numerical optimization và có cơ chế recombination tốt.

### PSO

PSO xem mỗi nghiệm như một particle bay trong không gian tìm kiếm. Particle bị kéo bởi kinh nghiệm tốt nhất của chính nó và global best của swarm.

### Simple ES

Simple ES tạo nghiệm mới bằng nhiễu Gaussian quanh nghiệm tốt. Tham số `sigma` quyết định bước nhảy:

```text
sigma nhỏ: khai thác tốt nhưng khám phá chậm
sigma lớn: khám phá nhanh nhưng khó tinh chỉnh nghiệm
```

### GA

GA dùng selection, mutation, crossover. Trong project:

- `ga-0.02`: bước mutation nhỏ.
- `ga-0.25`: bước vừa.
- `ga-0.5`: bước lớn.

Crossover giúp trộn hai nghiệm cha mẹ để sinh nghiệm mới.

---

## 25. Tại sao cần nhiều giá trị sigma?

Với GA và Simple ES, `sigma` điều khiển độ mạnh của mutation.

Ý nghĩa:

| Sigma | Ý nghĩa | Ưu điểm | Nhược điểm |
|---:|---|---|---|
| 0.02 | Bước nhỏ | Tinh chỉnh tốt gần optimum | Dễ đi chậm, khó thoát xa |
| 0.25 | Bước vừa | Cân bằng exploration/exploitation | Không luôn tối ưu mọi problem |
| 0.5 | Bước lớn | Khám phá mạnh hơn | Có thể dao động, khó hội tụ mịn |

Với PA-10, optimum nằm xa gốc, sigma lớn có thể giúp di chuyển nhanh hơn. Với Sphere hoặc Rastrigin gần gốc, sigma nhỏ có thể giúp tinh chỉnh tốt hơn.

---

## 26. Vì sao phải dùng seed?

EA có tính ngẫu nhiên ở nhiều bước:

- sinh population ban đầu;
- chọn parent;
- mutation;
- crossover;
- sampling vector mới;
- update swarm/distribution.

Do đó một lần chạy chưa đủ kết luận. Dùng 30 seeds giúp đánh giá:

- performance trung bình;
- độ ổn định;
- outlier;
- so sánh công bằng hơn giữa solver.

---

## 27. Những điểm khác với paper gốc

Project hiện tại mở rộng Scenario 1 so với mô tả paper ở điểm:

```text
Thêm Rosenbrock và Griewank.
```

Vì vậy:

```text
Paper gốc: 24 synthetic problems
Project hiện tại: 32 synthetic problems
```

Khi viết báo cáo, nên nói rõ:

```text
We follow the synthetic benchmark setup and extend it with Rosenbrock and Griewank for additional landscape diversity.
```

Hoặc tiếng Việt:

```text
Chúng tôi kế thừa thiết kế synthetic benchmark và mở rộng thêm Rosenbrock, Griewank để đa dạng hóa landscape.
```

---

## 28. Cách đọc kết quả cuối

Ví dụ một dòng trong `Scenario_1_final.csv` có dạng:

```text
seed = 1
problem = Sphere-20
solver_sigma = cmaEs
n.evals = 10008
best→genotype→size = 20
best→quality = 1.056820e-37
```

Diễn giải:

```text
Với seed 1, bài Sphere p=20, solver CMA-ES chạy khoảng 10008 evaluations.
Nghiệm tốt nhất có vector size 20.
Fitness tốt nhất gần 0, nghĩa là solver tìm được nghiệm rất sát optimum.
```

---

## 29. Kết luận thiết kế

Thực nghiệm 1 là một benchmark numerical optimization. Nó kiểm tra khả năng tự tìm nghiệm của các EA trên nhiều hàm synthetic.

Tóm tắt:

```text
Input:
  - hàm f(x)
  - số chiều p
  - solver EA
  - seed
  - budget 10,000 evaluations

EA làm:
  - tự sinh population nghiệm ban đầu
  - tính fitness cho từng nghiệm
  - chọn nghiệm tốt
  - tạo nghiệm mới bằng mutation/crossover/update
  - lặp lại đến khi hết budget

Output một run:
  - best_x
  - best_fitness

Output toàn bộ thực nghiệm:
  - Scenario_1.csv: log quá trình
  - Scenario_1_final.csv: kết quả cuối của 8640 runs

Thiết bị chạy:
  - Intel i7-12700H
  - 32 GB RAM
  - Windows 64-bit
  - GPU không phải yếu tố chính
```

---

## 30. Khuyến nghị cho báo cáo/slide

Nên trình bày thực nghiệm 1 theo thứ tự:

```text
1. Mục tiêu: so sánh EA trên synthetic functions
2. Input: f(x), p, solver, seed
3. Candidate solution: x ∈ R^p
4. EA process: initialize → evaluate → evolve → stop
5. Problems: Sphere, PA, CPA, Ackley, Rastrigin, Rosenbrock, Griewank
6. Solvers: CMA-ES, DE, PSO, ES, GA
7. Budget: 10,000 evaluations
8. Runs: 30 seeds × 32 problems × 9 solvers
9. Output: best quality, convergence, final CSV
10. Visualization: landscape, population movement, convergence curve, boxplot
```

Một câu nói ngắn gọn cho slide:

```text
In Scenario 1, each EA optimizes a synthetic objective function f(x) by automatically generating and evolving candidate vectors x ∈ R^p. Each run starts from random vectors sampled in [-1,1]^p, evaluates their fitness, repeatedly creates new candidates using algorithm-specific operators, and stops after 10,000 fitness evaluations. The final output is the best solution and its best fitness, logged across 30 seeds for statistical comparison.
```
