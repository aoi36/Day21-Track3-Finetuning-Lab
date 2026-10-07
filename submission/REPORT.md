# Lab 21 — Evaluation Report

**Họ tên**: chưa được cung cấp  **MSSV**: chưa được cung cấp  **Ngày chạy**: 2026-10-07  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: Tesla T4, CUDA sm_75, 14.6 GB khả dụng

> Báo cáo này ghi kết quả full evaluation từ lần chạy Colab được cung cấp. Khác với
> lượt smoke trước, lần này NB2/NB5 dùng đủ 50 ticket target và 15 câu regression.
> `results/` của lần chạy được tạo trong runtime Colab; các giá trị dưới đây được chép
> từ log và phần xác minh đã dán.

---

## 1. Setup

| | |
|---|---|
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 theo tier T4; p95 đo được là 98 token |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |
| Evaluation | Đầy đủ: 50 target, 15 regression; `EVAL_LIMIT` không đặt |
| Cấu hình LoRA đúng | `text-linear`, r=16, alpha=32, LR=1e-4 |

NB1 đề xuất `max_length=256`, nhưng lần chạy giữ giá trị tier T4 là 1024 để theo cấu
hình mặc định. Giá trị này lớn hơn nhiều so với p95=98 token, nên có thể gây lãng phí
padding; đây là điểm cần cân nhắc ở lần thử tiếp theo. T4 không hỗ trợ bf16, do đó log
xác nhận dùng fp16 với gradient scaling. Model có 32 layer, gồm 24 layer linear
attention và 8 layer full attention; adapter đặt trên 12 loại linear của text decoder.

**Template có giữ khối `<think>` không?** Có. NB1 báo “reasoning preserved”. Dataset
huấn luyện mặc định có câu trả lời JSON, không có reasoning trace thực để đưa vào loss.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token trong ví dụ proof NB1) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Trên tập train đã tokenize, NB3 ghi nhận 9,014/20,951 token được giám sát (43.0%).
Đoạn supervised trong ví dụ NB1:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Mask được dùng để huấn luyện là `assistant-only`; prompt người dùng không bị tính loss.
NB1 cũng minh họa đối chứng `everything`, nhưng chế độ đó không được sử dụng trong NB3.

---

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3286.1 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1042.2 |
| (c) LoRA fine-tune | 0.9700 | 0.6111 | 1.0000 | 1415.7 |

Baseline (b) mạnh hơn (a) rõ rệt trên target, format và latency. SHA prompt được đóng
băng là `719e74d3b6232053`; log và verifier xác nhận prompt (b) không bị sửa sau khi
đo. Fine-tune vượt target của (b) 0.2050, nhưng giảm regression 0.1800; vì thế tăng
độ chính xác ticket không đồng nghĩa cải thiện tổng thể mà không có tác dụng phụ.

---

## 4. Giải phẫu cấu hình (NB4)

| Run | Vị trí | r | Trainable params | LR | Train loss | Target (NB5 §4) | Thời gian train (s) | VRAM peak (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6267 | 0.970 | 406.2 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5369 | 0.970 | 261.4 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 | 391.7 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 458.6 | 3.86 |

Tất cả bốn run dùng cùng 30 optimizer steps. `attn_only` dùng rank 283 để ghép ngân
sách tham số với `correct`; chênh lệch trainable params là 8,192, thấp hơn 0.05%, nên
đây là so sánh placement với ngân sách gần như ngang nhau.

**4.1 — `attn_only` và `correct`.** Trên đủ 50 mẫu target, hai run hòa ở 0.970. Thứ
tự theo train loss khác: `attn_only` có loss thấp hơn, 0.5369 so với 0.6267, nhưng
không đạt target score cao hơn. Vì số tham số đã được ghép tương đương, kết quả không
cho thấy chỉ tăng rank là đủ để cải thiện chất lượng tác vụ. Với dữ liệu và bước chạy
này, placement khác nhau dẫn tới loss khác nhưng cùng target accuracy; kết quả không
ủng hộ tuyên bố một placement thắng placement kia.

**4.2 — `wrong_lr`.** Run này chỉ đổi LR từ 1e-4 xuống 1e-5. Loss cuối là 1.5702, cao
hơn đáng kể so với 0.6267 của `correct`; log huấn luyện cho thấy `wrong_lr` giảm chậm
hơn và vẫn ở 1.119 tại log step cuối, trong khi `correct` đã giảm rất sâu. Target và
format đều bằng 0.000 trên 50 mẫu. Nếu chỉ quan sát việc loss có giảm so với ban đầu,
có thể kết luận nhầm rằng mô hình đã học đủ; cần xem cả loss cuối, độ ổn định của quá
trình train và metric trên tác vụ thực tế.

**4.3 — `qlora`.** Peak VRAM giảm từ 8.78 GB xuống 3.86 GB, tiết kiệm 4.92 GB, khoảng
56%. Đổi lại, train loss tăng từ 0.6267 lên 0.7058 và target giảm từ 0.970 xuống 0.940;
latency cũng tăng từ 1415.7 lên 1794.2 ms/mẫu. Kết quả đầy đủ này ủng hộ khuyến nghị
thận trọng với QLoRA trên model này nếu GPU đủ chạy LoRA thường. Nếu giới hạn VRAM là
yếu tố quyết định, QLoRA vẫn là phương án đánh đổi có thể cân nhắc, chứ không phải
một lựa chọn không bao giờ dùng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.2050` · `regression Δ = -0.1800` · `valid_trace_rate = 0.0000`

Fine-tune tăng target score từ 0.7650 lên 0.9700 trên đủ 50 ticket và giữ format ở
1.000. Tuy nhiên, regression giảm từ 0.7911 xuống 0.6111 trên 15 câu hỏi kiến thức và
hướng dẫn phổ thông. Mức giảm 0.1800 vượt xa ngưỡng cho phép 0.020, nên cổng báo
`FAILED` dù mục tiêu phân loại ticket tăng mạnh. Đây là dấu hiệu rằng huấn luyện trên
225 ticket CSKH chuyên miền đã làm suy giảm một phần năng lực tổng quát đo bởi bộ
regression. `valid_trace_rate=0.0`; corpus không có trace suy luận thực nên chỉ số này
không phải mục tiêu được huấn luyện trong thí nghiệm. Kết quả thất bại không nên bị
che bằng cách nới ngưỡng hoặc bỏ regression set. Hướng tiếp theo hợp lý là thử thêm
1–5% replay examples đại diện cho năng lực chung vào tập train, giữ nguyên eval đã đóng
băng, rồi chạy lại và xem liệu mức regression được phục hồi mà target vẫn giữ cao hay
không.

---

## 6. Định tính — ticket tốt và ticket còn lỗi

Log NB5 in ba ca có target score thấp nhất và ba ca cao nhất cho fine-tune. Bảng dưới
ghi nhãn và dự đoán FT rút ra từ log (ca thấp nhất lệch ở urgency; ca cao nhất đạt đủ
4/4 trường). Log được cung cấp không có dự đoán riêng của baseline (b) theo từng ticket,
nên không khẳng định những ca này là thắng hoặc thua FT so với baseline ở cấp từng mẫu.
Muốn hoàn thiện phép đối chiếu per-ticket, cần xuất predictions của (b) cùng
`qualitative.json` từ runtime Colab.

| Eval index | Ticket (rút gọn) | Nhãn đúng (4 trường) | FT prediction | Kết quả FT |
|---:|---|---|---|---|
| 3 | Bình giữ nhiệt; chưa thấy tiền; khi nào tiện | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | `hoan_tien / trung_binh / bình giữ nhiệt / tich_cuc` | 3/4; urgency sai |
| 5 | Nồi chiên không dầu; thiếu phụ kiện; khi nào tiện | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | `san_pham_loi / trung_binh / nồi chiên không dầu / trung_tinh` | 3/4; urgency sai |
| 12 | Áo khoác gió; bị lỗi; khi nào tiện | `san_pham_loi / thap / áo khoác gió / tich_cuc` | `san_pham_loi / trung_binh / áo khoác gió / tich_cuc` | 3/4; urgency sai |
| 47 | Ốp lưng điện thoại; shipper không gọi; hỗ trợ tốt | `van_chuyen / thap / ốp lưng điện thoại / tich_cuc` | Trùng nhãn trên cả 4 trường | 4/4 |
| 48 | Ốp lưng điện thoại; hỏi giá | `hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh` | Trùng nhãn trên cả 4 trường | 4/4 |

Trong ba lỗi FT thấp nhất được log, urgency đều bị dự đoán là `trung_binh` trong khi
nhãn là `thap`, dù nội dung có câu “khi nào tiện”. Đây là một mẫu lỗi cụ thể đáng kiểm
tra thêm bằng dữ liệu huấn luyện cân bằng hơn về cách diễn đạt mức khẩn cấp. Các ca
điểm 1.0 cho thấy FT xử lý chính xác một số ticket vận chuyển, hỏi thông tin và lỗi
sản phẩm, nhưng chưa đủ để khẳng định thắng baseline từng mẫu.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi chưa deploy adapter này. Full evaluation cho thấy LoRA `correct` rất
tốt trên ticket CSKH: target đạt 0.970 so với 0.765 của baseline dùng optimized prompt,
và cả hai đều tạo JSON đúng format trên toàn bộ 50 ticket. Nhưng regression giảm 0.180
trên 15 câu hỏi tổng quát, khiến gate thất bại so với ngưỡng suy giảm 0.020. Đây không
phải một lỗi có thể bỏ qua chỉ vì target tăng; một hệ thống CSKH vẫn có thể cần trả lời
các câu hỏi ngoài khuôn khổ ticket và không nên mất năng lực đó. Trước lần train tiếp
theo, tôi sẽ bổ sung một lượng nhỏ replay data tổng quát độc lập, bắt đầu ở mức 1–5%
theo khuyến nghị lab, đồng thời giữ nguyên tập eval và prompt đã đóng băng. Sau đó tôi
sẽ kiểm tra đồng thời target, regression, format và latency; chỉ xem xét deploy nếu
regression trở lại trong ngưỡng cho phép mà target vẫn có lợi ích. Trong các đối chứng,
LR 1e-5 cho kết quả tệ hơn rõ rệt; `attn_only` hòa với `correct` ở target dù loss thấp
hơn; QLoRA tiết kiệm VRAM nhưng giảm nhẹ chất lượng và chậm hơn. Mask đúng bảo đảm thí
nghiệm học trên câu trả lời, nhưng replay data có vẻ là đòn bẩy cần thử tiếp để xử lý
tradeoff giữa chuyên môn ticket và năng lực tổng quát.

**Ba điều tôi học được:**
1. Baseline prompt tốt đã đạt target 0.765; cần vượt một baseline tử tế, không phải chỉ
   model với prompt ngây thơ.
2. Target tăng không đảm bảo model tốt hơn nói chung: regression giảm 0.180 làm gate
   thất bại dù target tăng 0.205.
3. Train loss thấp hơn không nhất thiết tạo target score cao hơn: `attn_only` có loss
   thấp hơn `correct`, nhưng cả hai cùng target 0.970.

**Nếu có thêm 2 giờ nữa, tôi sẽ:** tạo tập replay tổng quát nhỏ, độc lập với 15 câu
regression đánh giá; thử tỷ lệ replay tăng dần trong khoảng 1–5%; train lại với cùng
seed, số step và base model; sau đó chạy nguyên bộ eval đã đóng băng. Tôi cũng sẽ lưu
predictions của baseline và fine-tune theo từng ticket để phân tích chính xác ca thắng
và ca thua.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap — chưa chạy
- [ ] B2 dataset miền riêng — không; dùng corpus mặc định
- [ ] B3 reasoning-trace collapse — không; dùng `assistant-only`
- [ ] B4 quét rank có kiểm soát — không
- [ ] B5 HuggingFace Hub — chưa đăng
