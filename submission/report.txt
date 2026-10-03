Tôi dùng AWS, region us-east-1, CPU t3.medium; source commit 55539f67d7c78b43afe334a2ec3271c4bfdbbe2d.
Dataset có 284,807 giao dịch; chia train/validation/test 60/20/20, stratify theo Class, seed 42.
Load dữ liệu 2.31 giây; training 2.46 giây; best iteration 2.
Test AUC 0.9409, Accuracy 98.98%, F1 0.2331, Precision 13.39%, Recall 89.80%.
Recall cao nhưng Precision thấp cho thấy nhiều cảnh báo nhầm; Accuracy cần được đánh giá thận trọng do dữ liệu mất cân bằng.
Latency 1.16 ms/dòng (trung bình 100 lần sau warm-up); throughput 669,552 dòng/giây (trung bình 20 batch, mỗi batch 1.000 dòng, 2 luồng). Mô hình chỉ có 2 vòng boosting.
Ảnh sau benchmark: top ghi CPU 3,2% user và 96,8% idle; free -h ghi RAM tổng 3,7 GiB, dùng 250 MiB; interface ens5 ghi RX 283.464.196 byte, TX 1.114.632 byte. Xem ảnh 02_cpu, 03_ram, 04_network.
Billing tháng 10/2026 tại thời điểm chụp hiển thị USD 0.00, chưa có EC2/NAT Gateway; ảnh billing.png chưa thể hiện chi phí cuối cùng.
Kết quả đã tải về laptop; người thực hiện báo destroy thành công. Kiểm tra lúc 2026-10-04T01:27:41+07:00 (giờ Việt Nam): terraform state list trống; bằng chứng cleanup_verification.txt. Chưa xác minh độc lập qua Console.
