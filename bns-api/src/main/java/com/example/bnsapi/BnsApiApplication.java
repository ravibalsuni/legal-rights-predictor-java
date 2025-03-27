package com.example.bnsapi;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;

import javax.persistence.*;
import java.io.InputStream;
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.Arrays;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.Encoding;

@SpringBootApplication
public class BnsApiApplication {
    public static void main(String[]args) {
        SpringApplication.run(BnsApiApplication.class, args);
    }
}

@RestController
@RequestMapping("/api/bns")
class BnsController {
    @Autowired
    private BnsService bnsService;

    @PostMapping("/predict")
    public List<BnsSection> predict(@RequestBody String query) {
        return bnsService.findRelevantSections(query);
    }
}

@Service
class BnsService implements CommandLineRunner {
    private static final Logger logger = Logger.getLogger(BnsService.class.getName());
    @Autowired
    private BnsRepository bnsRepository;
    private final Map<Long, float[]> embeddingCache = new HashMap<>();
    private HuggingFaceTokenizer tokenizer;

    public BnsService() {
        try {
            tokenizer = HuggingFaceTokenizer.newInstance("bert-base-uncased");
        } catch (Exception e) {
            logger.severe("Error loading tokenizer: " + e.getMessage());
        }
    }

    @Override
    public void run(String... args) {
        loadBnsData();
        List<BnsSection> sections = bnsRepository.findAll();
        for (BnsSection section : sections) {
            if (section.getEmbedding() == null || section.getEmbedding().length == 0) {
                float[] embedding = computeBertEmbedding(section.getTitle() + " " + section.getDescription());
                section.setEmbedding(embedding);
                bnsRepository.save(section);
            }
            embeddingCache.put(section.getId(), section.getEmbedding());
        }
    }

    private void loadBnsData() {
        try (InputStream inputStream = getClass().getClassLoader().getResourceAsStream("BNS 2K24 BY Dhiraj.xlsx");
             Workbook workbook = new XSSFWorkbook(inputStream)) {
            Sheet sheet = workbook.getSheetAt(0);
            for (Row row : sheet) {
                if (row.getRowNum() == 0) continue;
                String sectionNo = row.getCell(0).getStringCellValue();
                String title = row.getCell(1).getStringCellValue();
                String description = row.getCell(2).getStringCellValue();
                String punishment = row.getCell(3).getStringCellValue();
                BnsSection section = new BnsSection(sectionNo, title, description, punishment);
                bnsRepository.save(section);
            }
        } catch (Exception e) {
            logger.severe("Error loading BNS data: " + e.getMessage());
        }
    }

    public List<BnsSection> findRelevantSections(String query) {
        float[] queryEmbedding = computeBertEmbedding(query);
        return embeddingCache.entrySet().stream()
                .sorted(Comparator.comparingDouble(entry -> -cosineSimilarity(queryEmbedding, entry.getValue())))
                .limit(4)
                .map(entry -> bnsRepository.findById(entry.getKey()).orElse(null))
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    private float[] computeBertEmbedding(String text) {
        try {
            Encoding encoding = tokenizer.encode(text, String.valueOf(128));
            long[] inputIds = Arrays.copyOf(encoding.getIds(), 128);
            float[] embedding = new float[inputIds.length];
            for (int i = 0; i < inputIds.length; i++) {
                embedding[i] = (float) inputIds[i];
            }
            return embedding;
        } catch (Exception e) {
            logger.severe("Error computing BERT embedding: " + e.getMessage());
            return new float[128];
        }
    }

    private double cosineSimilarity(float[] vec1, float[] vec2) {
        double dotProduct = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            normA += Math.pow(vec1[i], 2);
            normB += Math.pow(vec2[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

interface BnsRepository extends JpaRepository<BnsSection, Long> {}

@Entity
class BnsSection {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String sectionNo;
    private String title;
    @Column(columnDefinition = "TEXT")
    private String description;
    private String punishment;

    @Lob
    @Column(columnDefinition = "TEXT")
    private String embedding;

    public BnsSection() {}

    public BnsSection(String sectionNo, String title, String description, String punishment) {
        this.sectionNo = sectionNo;
        this.title = title;
        this.description = description;
        this.punishment = punishment;
    }

    public Long getId() { return id; }
    public String getSectionNo() { return sectionNo; }
    public String getTitle() { return title; }
    public String getDescription() { return description; }
    public String getPunishment() { return punishment; }

    public float[] getEmbedding() {
        if (embedding == null || embedding.isEmpty()) {
            return new float[128];
        } else {
            String[] values = embedding.replace("[", "").replace("]", "").split(",");
            float[] floats = new float[values.length];

            for (int i = 0; i < values.length; i++) {
                floats[i] = Float.parseFloat(values[i].trim());
            }
            return floats;
        }
    }

    public void setEmbedding(float[] embedding) {
        this.embedding = Arrays.toString(embedding);
    }
}