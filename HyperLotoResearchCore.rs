// HyperLoto Research Core (Rust / Native-ready)
// -------------------------------------------------
// 目的:
// - Android / Linux / Windows / macOS で共有できる研究用コア
// - MonteCarlo, GA, Gap分析, 共起グラフ, ポートフォリオ生成を1ファイルで概観
//
// 推奨 Cargo.toml 依存関係:
// [dependencies]
// rand = "0.8"
// rand_distr = "0.4"
// rayon = "1.10"
// serde = { version = "1", features = ["derive"] }
// serde_json = "1"
// clap = { version = "4", features = ["derive"] }
// petgraph = "0.6"
// anyhow = "1"
// csv = "1"
//
// これは単体でも読みやすいように1ファイルへ集約しています。

use anyhow::{Context, Result};
use clap::Parser;
use petgraph::graph::{NodeIndex, UnGraph};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Write};

const SET_NAMES: [&str; 4] = ["C", "D", "E", "H"];
const SET_BIAS: [(&str, &[(u8, f64)]); 4] = [
    ("C", &[(4,26.0),(9,21.0),(35,21.0),(13,20.0),(8,20.0),(34,20.0),(28,19.0),(10,18.0),(2,18.0),(6,17.0),(26,17.0),(36,17.0),(11,17.0),(23,16.0),(27,15.0),(18,15.0),(14,15.0),(24,15.0)]),
    ("D", &[(8,23.0),(21,21.0),(27,21.0),(13,21.0),(9,20.0),(36,20.0),(28,20.0),(30,20.0),(23,19.0),(4,19.0),(31,18.0),(14,18.0),(24,17.0),(1,17.0),(2,16.0),(18,16.0),(15,16.0)]),
    ("E", &[(32,23.0),(17,22.0),(34,22.0),(36,22.0),(1,21.0),(11,21.0),(19,21.0),(30,20.0),(24,19.0),(5,18.0),(33,18.0),(10,18.0),(6,18.0),(25,17.0),(31,17.0),(7,17.0),(37,17.0)]),
    ("H", &[(12,19.0),(32,19.0),(20,18.0),(28,18.0),(18,18.0),(37,18.0),(22,17.0),(9,17.0),(15,16.0),(27,16.0),(4,15.0),(29,15.0),(10,15.0),(26,14.0),(24,14.0),(35,14.0),(5,14.0)]),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Draw {
    pub draw_number: u32,
    pub date: String,
    pub numbers: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRow {
    pub number: u8,
    pub score: f64,
    pub long_term: f64,
    pub recent: f64,
    pub gap: f64,
    pub set_bias: f64,
    pub pair_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComboScore {
    pub combo: Vec<u8>,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    pub top_combos: Vec<(Vec<u8>, usize)>,
    pub top_numbers: Vec<(u8, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetEstimate {
    pub best: String,
    pub ranking: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchReport {
    pub loto_type: String,
    pub prediction: Vec<u8>,
    pub set_estimate: Option<SetEstimate>,
    pub monte_carlo: MonteCarloResult,
    pub ga_top: Vec<ComboScore>,
    pub portfolio: Vec<ComboScore>,
    pub score_table: Vec<ScoreRow>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub loto_type: LotoType,
    pub recent_window: usize,
    pub monte_iterations: usize,
    pub ga_population: usize,
    pub ga_generations: usize,
    pub portfolio_size: usize,
    pub use_auto_set: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LotoType {
    Loto6,
    Loto7,
}

impl LotoType {
    pub fn max_num(self) -> u8 {
        match self {
            Self::Loto6 => 43,
            Self::Loto7 => 37,
        }
    }
    pub fn pick_count(self) -> usize {
        match self {
            Self::Loto6 => 6,
            Self::Loto7 => 7,
        }
    }
    pub fn band_ranges(self) -> Vec<(u8, u8)> {
        match self {
            Self::Loto6 => vec![(1,11),(12,22),(23,33),(34,43)],
            Self::Loto7 => vec![(1,9),(10,18),(19,27),(28,37)],
        }
    }
    pub fn band_targets(self) -> Vec<usize> {
        match self {
            Self::Loto6 => vec![2,2,1,1],
            Self::Loto7 => vec![2,2,2,1],
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loto6 => "loto6",
            Self::Loto7 => "loto7",
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about = "HyperLoto Research Core")]
struct Cli {
    #[arg(long)]
    input_csv: String,
    #[arg(long, default_value = "loto7")]
    loto: String,
    #[arg(long, default_value_t = 20)]
    recent_window: usize,
    #[arg(long, default_value_t = 10000)]
    monte: usize,
    #[arg(long, default_value_t = 120)]
    ga_population: usize,
    #[arg(long, default_value_t = 50)]
    ga_generations: usize,
    #[arg(long, default_value_t = 100)]
    portfolio: usize,
    #[arg(long, default_value_t = true)]
    auto_set: bool,
    #[arg(long, default_value = "hyperloto_report.json")]
    output_json: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let loto_type = if cli.loto.eq_ignore_ascii_case("loto6") {
        LotoType::Loto6
    } else {
        LotoType::Loto7
    };

    let config = Config {
        loto_type,
        recent_window: cli.recent_window,
        monte_iterations: cli.monte,
        ga_population: cli.ga_population,
        ga_generations: cli.ga_generations,
        portfolio_size: cli.portfolio,
        use_auto_set: cli.auto_set,
    };

    let draws = load_draws(&cli.input_csv, loto_type)?;
    let report = run_research_pipeline(&draws, &config)?;

    let mut file = File::create(&cli.output_json)
        .with_context(|| format!("failed to create {}", cli.output_json))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;

    println!("done: {}", cli.output_json);
    Ok(())
}

pub fn run_research_pipeline(draws: &[Draw], config: &Config) -> Result<ResearchReport> {
    let set_estimate = if config.loto_type == LotoType::Loto7 && config.use_auto_set {
        Some(estimate_set(draws, config))
    } else {
        None
    };

    let pair_matrix = build_pair_matrix(draws, config.loto_type);
    let _graph = build_pair_graph(&pair_matrix, config.loto_type);
    let score_table = compute_score_table(draws, config, set_estimate.as_ref(), &pair_matrix);
    let prediction = build_prediction(&score_table, config);
    let monte_carlo = monte_carlo_search(&score_table, config);
    let ga_top = genetic_search(&score_table, config);
    let portfolio = build_portfolio(&score_table, &ga_top, config);

    Ok(ResearchReport {
        loto_type: config.loto_type.as_str().to_string(),
        prediction,
        set_estimate,
        monte_carlo,
        ga_top,
        portfolio,
        score_table,
    })
}

pub fn load_draws(path: &str, loto_type: LotoType) -> Result<Vec<Draw>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path))?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let pick_count = loto_type.pick_count();
    let max_num = loto_type.max_num();
    let mut draws = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 + pick_count {
            continue;
        }
        let draw_number: u32 = record.get(0).unwrap_or("0").parse().unwrap_or(0);
        let date = record.get(1).unwrap_or("").to_string();
        if draw_number == 0 || date.is_empty() {
            continue;
        }
        let mut set = BTreeSet::new();
        for idx in 2..record.len() {
            if set.len() >= pick_count {
                break;
            }
            let n: u8 = record.get(idx).unwrap_or("0").parse().unwrap_or(0);
            if n >= 1 && n <= max_num {
                set.insert(n);
            }
        }
        if set.len() == pick_count {
            draws.push(Draw {
                draw_number,
                date,
                numbers: set.into_iter().collect(),
            });
        }
    }

    draws.reverse();
    Ok(draws)
}

fn normalize(values: &[f64]) -> Vec<f64> {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min.is_finite() || !max.is_finite() || (max - min).abs() < f64::EPSILON {
        return vec![0.5; values.len()];
    }
    values.iter().map(|v| (v - min) / (max - min)).collect()
}

fn estimate_set(draws: &[Draw], config: &Config) -> SetEstimate {
    let freq = frequency_counts(draws, config.loto_type.max_num());
    let mut pairs: Vec<(u8, usize)> = freq.iter().map(|(k, v)| (*k, *v)).collect();
    pairs.sort_by_key(|(_, c)| usize::MAX - *c);
    let rank_map: HashMap<u8, usize> = pairs.into_iter().enumerate().map(|(i, (n, _))| (n, i + 1)).collect();

    let mut ranking = Vec::new();
    for (name, entries) in SET_BIAS.iter() {
        let mut score = 0.0;
        for (n, c) in *entries {
            let rank = *rank_map.get(n).unwrap_or(&37) as f64;
            score += c / (rank + 4.0);
        }
        score /= entries.len() as f64;
        ranking.push((name.to_string(), score));
    }
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    SetEstimate {
        best: ranking[0].0.clone(),
        ranking,
    }
}

fn frequency_counts(draws: &[Draw], max_num: u8) -> BTreeMap<u8, usize> {
    let mut map = BTreeMap::new();
    for n in 1..=max_num {
        map.insert(n, 0);
    }
    for draw in draws {
        for &n in &draw.numbers {
            *map.entry(n).or_insert(0) += 1;
        }
    }
    map
}

fn build_pair_matrix(draws: &[Draw], loto_type: LotoType) -> Vec<Vec<usize>> {
    let size = loto_type.max_num() as usize + 1;
    let mut matrix = vec![vec![0; size]; size];
    for draw in draws {
        for i in 0..draw.numbers.len() {
            for j in (i + 1)..draw.numbers.len() {
                let a = draw.numbers[i] as usize;
                let b = draw.numbers[j] as usize;
                matrix[a][b] += 1;
                matrix[b][a] += 1;
            }
        }
    }
    matrix
}

fn build_pair_graph(pair_matrix: &[Vec<usize>], loto_type: LotoType) -> UnGraph<u8, usize> {
    let mut g = UnGraph::<u8, usize>::new_undirected();
    let mut nodes = HashMap::<u8, NodeIndex>::new();
    for n in 1..=loto_type.max_num() {
        let idx = g.add_node(n);
        nodes.insert(n, idx);
    }
    for a in 1..=loto_type.max_num() as usize {
        for b in (a + 1)..=loto_type.max_num() as usize {
            let w = pair_matrix[a][b];
            if w > 0 {
                g.add_edge(nodes[&(a as u8)], nodes[&(b as u8)], w);
            }
        }
    }
    g
}

fn compute_score_table(
    draws: &[Draw],
    config: &Config,
    set_estimate: Option<&SetEstimate>,
    pair_matrix: &[Vec<usize>],
) -> Vec<ScoreRow> {
    let max_num = config.loto_type.max_num();
    let freq = frequency_counts(draws, max_num);
    let recent_draws = &draws[..draws.len().min(config.recent_window)];
    let recent_freq = frequency_counts(recent_draws, max_num);

    let long_vec: Vec<f64> = (1..=max_num).map(|n| *freq.get(&n).unwrap_or(&0) as f64).collect();
    let recent_vec: Vec<f64> = (1..=max_num).map(|n| *recent_freq.get(&n).unwrap_or(&0) as f64).collect();
    let gap_vec: Vec<f64> = (1..=max_num)
        .map(|n| {
            draws.iter()
                .position(|d| d.numbers.contains(&n))
                .map(|p| p as f64)
                .unwrap_or(draws.len() as f64 + 10.0)
        })
        .collect();
    let pair_strength_raw: Vec<f64> = (1..=max_num)
        .map(|n| pair_matrix[n as usize].iter().map(|v| *v as f64).sum())
        .collect();

    let set_bias_raw: Vec<f64> = if config.loto_type == LotoType::Loto7 {
        let chosen = set_estimate.map(|s| s.best.as_str()).unwrap_or("C");
        let bias_map: HashMap<u8, f64> = SET_BIAS
            .iter()
            .find(|(name, _)| *name == chosen)
            .map(|(_, entries)| entries.iter().copied().collect())
            .unwrap_or_default();
        (1..=max_num).map(|n| *bias_map.get(&n).unwrap_or(&0.0)).collect()
    } else {
        vec![0.0; max_num as usize]
    };

    let long_norm = normalize(&long_vec);
    let recent_norm = normalize(&recent_vec);
    let gap_norm = normalize(&gap_vec);
    let pair_norm = normalize(&pair_strength_raw);
    let set_norm = normalize(&set_bias_raw);

    let mut rows = Vec::new();
    for i in 0..max_num as usize {
        let number = (i + 1) as u8;
        let score = long_norm[i] * 0.34
            + recent_norm[i] * 0.24
            + gap_norm[i] * 0.16
            + set_norm[i] * 0.10
            + pair_norm[i] * 0.16;
        rows.push(ScoreRow {
            number,
            score,
            long_term: long_norm[i],
            recent: recent_norm[i],
            gap: gap_norm[i],
            set_bias: set_norm[i],
            pair_strength: pair_norm[i],
        });
    }
    rows.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    rows
}

fn build_prediction(score_table: &[ScoreRow], config: &Config) -> Vec<u8> {
    let ranges = config.loto_type.band_ranges();
    let targets = config.loto_type.band_targets();
    let mut selected = Vec::new();
    let mut used = HashSet::new();

    for ((lo, hi), need) in ranges.into_iter().zip(targets) {
        let candidates: Vec<u8> = score_table
            .iter()
            .filter(|r| r.number >= lo && r.number <= hi)
            .map(|r| r.number)
            .take(12)
            .collect();
        for n in candidates.into_iter().take(need) {
            if used.insert(n) {
                selected.push(n);
            }
        }
    }

    for row in score_table {
        if selected.len() >= config.loto_type.pick_count() {
            break;
        }
        if used.insert(row.number) {
            selected.push(row.number);
        }
    }

    selected.sort_unstable();
    selected
}

fn combo_score(combo: &[u8], score_table: &[ScoreRow], config: &Config) -> f64 {
    let score_map: HashMap<u8, f64> = score_table.iter().map(|r| (r.number, r.score)).collect();
    let sum_score: f64 = combo.iter().map(|n| score_map.get(n).copied().unwrap_or(0.0)).sum();
    let odd = combo.iter().filter(|n| **n % 2 == 1).count();
    let even = combo.len() - odd;
    let odd_even_penalty = if (odd as i32 - even as i32).abs() > 3 { 0.25 } else { 0.0 };
    let ranges = config.loto_type.band_ranges();
    let targets = config.loto_type.band_targets();
    let mut band_penalty = 0.0;
    for ((lo, hi), target) in ranges.into_iter().zip(targets) {
        let count = combo.iter().filter(|n| **n >= lo && **n <= hi).count();
        band_penalty += ((count as i32 - target as i32).abs() as f64) * 0.08;
    }
    sum_score - odd_even_penalty - band_penalty
}

fn random_combo_from_top(rng: &mut impl Rng, score_table: &[ScoreRow], config: &Config, top_n: usize) -> Vec<u8> {
    let pool: Vec<u8> = score_table.iter().take(top_n).map(|r| r.number).collect();
    let mut set = BTreeSet::new();
    while set.len() < config.loto_type.pick_count() {
        let n = pool[rng.gen_range(0..pool.len())];
        set.insert(n);
    }
    set.into_iter().collect()
}

fn monte_carlo_search(score_table: &[ScoreRow], config: &Config) -> MonteCarloResult {
    let iterations = config.monte_iterations;
    let top_n = if config.loto_type == LotoType::Loto7 { 18 } else { 20 };

    let combos: Vec<Vec<u8>> = (0..iterations)
        .into_par_iter()
        .map_init(rand::thread_rng, |rng, _| random_combo_from_top(rng, score_table, config, top_n))
        .collect();

    let mut combo_counts: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut number_counts: HashMap<u8, usize> = (1..=config.loto_type.max_num()).map(|n| (n, 0)).collect();

    for combo in combos {
        *combo_counts.entry(combo.clone()).or_insert(0) += 1;
        for n in combo {
            *number_counts.entry(n).or_insert(0) += 1;
        }
    }

    let mut top_combos: Vec<(Vec<u8>, usize)> = combo_counts.into_iter().collect();
    top_combos.sort_by_key(|(_, c)| usize::MAX - *c);
    top_combos.truncate(10);

    let mut top_numbers: Vec<(u8, usize)> = number_counts.into_iter().collect();
    top_numbers.sort_by_key(|(_, c)| usize::MAX - *c);
    top_numbers.truncate(10);

    MonteCarloResult { top_combos, top_numbers }
}

fn genetic_search(score_table: &[ScoreRow], config: &Config) -> Vec<ComboScore> {
    let top_n = if config.loto_type == LotoType::Loto7 { 22 } else { 24 };
    let mut rng = rand::thread_rng();
    let mut population: Vec<Vec<u8>> = (0..config.ga_population)
        .map(|_| random_combo_from_top(&mut rng, score_table, config, top_n))
        .collect();

    for _ in 0..config.ga_generations {
        let mut ranked: Vec<ComboScore> = population
            .par_iter()
            .map(|combo| ComboScore { combo: combo.clone(), score: combo_score(combo, score_table, config) })
            .collect();
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let elites: Vec<Vec<u8>> = ranked.iter().take((config.ga_population / 5).max(8)).map(|c| c.combo.clone()).collect();
        let mut next = elites.clone();

        while next.len() < config.ga_population {
            let a = &elites[rng.gen_range(0..elites.len())];
            let b = &elites[rng.gen_range(0..elites.len())];
            let mut merged: Vec<u8> = a.iter().take(a.len() / 2).copied().chain(b.iter().skip(b.len() / 2).copied()).collect();
            merged.sort_unstable();
            merged.dedup();
            while merged.len() < config.loto_type.pick_count() {
                let extra = score_table[rng.gen_range(0..top_n)].number;
                if !merged.contains(&extra) {
                    merged.push(extra);
                }
            }
            merged.truncate(config.loto_type.pick_count());
            merged.sort_unstable();
            next.push(merged);
        }
        population = next;
    }

    let mut final_ranked: Vec<ComboScore> = population
        .par_iter()
        .map(|combo| ComboScore { combo: combo.clone(), score: combo_score(combo, score_table, config) })
        .collect();
    final_ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut uniq = BTreeMap::<Vec<u8>, ComboScore>::new();
    for item in final_ranked {
        uniq.entry(item.combo.clone()).or_insert(item);
    }

    uniq.into_values().take(20).collect()
}

fn build_portfolio(score_table: &[ScoreRow], ga_top: &[ComboScore], config: &Config) -> Vec<ComboScore> {
    let mut results = Vec::new();
    let mut seen = HashSet::<Vec<u8>>::new();

    for item in ga_top {
        if seen.insert(item.combo.clone()) {
            results.push(item.clone());
        }
        if results.len() >= config.portfolio_size {
            return results;
        }
    }

    let top_n = if config.loto_type == LotoType::Loto7 { 22 } else { 24 };
    let mut rng = rand::thread_rng();
    while results.len() < config.portfolio_size {
        let combo = random_combo_from_top(&mut rng, score_table, config, top_n);
        if seen.insert(combo.clone()) {
            let score = combo_score(&combo, score_table, config);
            results.push(ComboScore { combo, score });
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results
}
