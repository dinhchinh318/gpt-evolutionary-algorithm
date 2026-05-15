#!/usr/bin/env bash
set -euo pipefail
# Chạy trong thư mục jgea-main đã giải nén từ jgea-main.zip.
# Yêu cầu: Java 21 và Maven 3.9.11+.

mvn -DskipTests clean package
JAR="io.github.ericmedvet.jgea.experimenter/target/jgea.experimenter-2.8.1-jar-with-dependencies.jar"
java -Xmx24g -jar "$JAR" -f scenario3_quick_test.txt
# Sau khi quick test chạy OK, bỏ comment dòng dưới để chạy bản giống paper:
# java -Xmx24g -jar "$JAR" -f scenario3_paper_like.txt
