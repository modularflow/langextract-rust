//! Visualization utilities for annotated documents.

use crate::{data::AnnotatedDocument, exceptions::LangExtractResult};
use crate::pipeline::PipelineResult;
use serde_json::{json, Value};
use std::collections::HashMap;
use crate::Extraction;
/// Export format options for visualization
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum ExportFormat {
    /// Simple text format (existing functionality)
    Text,
    /// Rich HTML with highlighting and interactivity
    Html,
    /// Structured markdown with summaries
    Markdown,
    /// Raw JSON export for analysis
    Json,
    /// CSV export for spreadsheet analysis
    Csv,
}

/// Configuration for visualization exports
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Export format to use
    pub format: ExportFormat,
    /// Show character intervals in output
    pub show_char_intervals: bool,
    /// Include original text in export
    pub include_text: bool,
    /// Highlight extractions in text (for HTML/Markdown)
    pub highlight_extractions: bool,
    /// Include extraction statistics
    pub include_statistics: bool,
    /// Custom CSS for HTML export
    pub custom_css: Option<String>,
    /// Title for the export
    pub title: Option<String>,
    /// Aggregate highlights across pipeline steps (pipeline HTML export)
    pub aggregate_pipeline_highlights: bool,
    /// Expand nested JSON extraction_text into atomic extractions when possible
    pub expand_nested_json: bool,
    /// Allow overlapping highlights (layered rendering)
    pub allow_overlapping_highlights: bool,
    /// Show legend for pipeline steps/colors
    pub show_pipeline_legend: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Text,
            show_char_intervals: false,
            include_text: true,
            highlight_extractions: true,
            include_statistics: true,
            custom_css: None,
            title: None,
            aggregate_pipeline_highlights: false,
            expand_nested_json: false,
            allow_overlapping_highlights: false,
            show_pipeline_legend: true,
        }
    }
}

/// Export an annotated document in the specified format
pub fn export_document(
    annotated_document: &AnnotatedDocument,
    config: &ExportConfig,
) -> LangExtractResult<String> {
    match config.format {
        ExportFormat::Text => visualize_text(annotated_document, config.show_char_intervals),
        ExportFormat::Html => export_html(annotated_document, config),
        ExportFormat::Markdown => export_markdown(annotated_document, config),
        ExportFormat::Json => export_json(annotated_document, config),
        ExportFormat::Csv => export_csv(annotated_document, config),
    }
}

/// Export a pipeline result as rich HTML with layered highlights per step
pub fn export_pipeline_html(
    pipeline_result: &PipelineResult,
    original_text: &str,
    config: &ExportConfig,
) -> LangExtractResult<String> {
    let title = config.title.as_deref().unwrap_or("LangExtract Pipeline Results");

    // Build layered spans from pipeline results, remapping to absolute intervals if needed
    let mut spans: Vec<LayeredSpan> = build_layered_spans(pipeline_result, original_text, config.expand_nested_json);
    spans.sort_by_key(|s| (s.start, s.end));

    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n");
    html.push_str("<html lang=\"en\">\n");
    html.push_str("<head>\n");
    html.push_str("    <meta charset=\"UTF-8\">\n");
    html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    html.push_str(&format!("    <title>{}</title>\n", title));
    html.push_str("    <style>\n");
    html.push_str("        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #334155; }\n");
    html.push_str("        .container { background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); overflow: hidden; }\n");
    html.push_str("        .header { background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%); color: white; padding: 30px; text-align: center; }\n");
    html.push_str("        .header h1 { margin: 0; font-size: 2.2em; font-weight: 400; }\n");
    html.push_str("        .content { padding: 30px; }\n");
    html.push_str("        .section { margin-bottom: 32px; }\n");
    html.push_str("        .section h2 { color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 16px; }\n");
    html.push_str("        .document-text { background: #f1f5f9; border-radius: 8px; padding: 16px; font-family: 'Monaco', 'Menlo', monospace; line-height: 1.6; white-space: pre-wrap; position: relative; }\n");
    html.push_str("        .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }\n");
    html.push_str("        .legend-item { display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; }\n");
    html.push_str("        .badge { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }\n");
    html.push_str("        .extraction-highlight { border-radius: 3px; padding: 1px 2px; cursor: pointer; }\n");
    html.push_str("        .step-0 { background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4); }\n");
    html.push_str("        .step-1 { background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); }\n");
    html.push_str("        .step-2 { background: rgba(234, 179, 8, 0.2); border: 1px solid rgba(234, 179, 8, 0.5); }\n");
    html.push_str("        .step-3 { background: rgba(244, 63, 94, 0.2); border: 1px solid rgba(244, 63, 94, 0.4); }\n");
    html.push_str("        .step-4 { background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4); }\n");
    html.push_str("    </style>\n");
    html.push_str("</head>\n");
    html.push_str("<body>\n");
    html.push_str("    <div class=\"container\">\n");
    html.push_str("        <div class=\"header\">\n");
    html.push_str(&format!("            <h1>{}</h1>\n", title));
    html.push_str("        </div>\n");
    html.push_str("        <div class=\"content\">\n");
    html.push_str("            <div class=\"section\">\n");
    html.push_str("                <h2>Document Text</h2>\n");
    if config.show_pipeline_legend {
        html.push_str(&build_legend_html(pipeline_result));
    }
    html.push_str("                <div class=\"document-text\">");
    html.push_str(&highlight_text_html_with_layers(original_text, &spans, config.allow_overlapping_highlights)?);
    html.push_str("</div>\n");
    html.push_str("            </div>\n");
    html.push_str("            <div class=\"section\">\n");
    html.push_str("                <h2>Extractions by Step</h2>\n");
    html.push_str("                <div>\n");
    html.push_str(&build_extractions_list_html(&spans));
    html.push_str("                </div>\n");
    html.push_str("            </div>\n");
    html.push_str("        </div>\n");
    html.push_str("    </div>\n");
    html.push_str("</body>\n");
    html.push_str("</html>\n");

    Ok(html)
}

/// Export a flattened JSON view of pipeline results (one item per atomic extraction)
pub fn export_pipeline_flattened_json(
    pipeline_result: &PipelineResult,
    original_text: &str,
    expand_nested_json: bool,
) -> LangExtractResult<String> {
    // Helper to push one flattened item
    fn push_item(
        items: &mut Vec<Value>,
        class_name: &str,
        text: &str,
        step_id: &str,
        step_name: &str,
        start: Option<usize>,
        end: Option<usize>,
        parent_attrs: Option<&std::collections::HashMap<String, Value>>,
    ) {
        let mut obj = serde_json::Map::new();
        obj.insert("extraction_class".to_string(), Value::String(class_name.to_string()));
        obj.insert("extraction_text".to_string(), Value::String(text.to_string()));
        obj.insert("step_id".to_string(), Value::String(step_id.to_string()));
        obj.insert("step_name".to_string(), Value::String(step_name.to_string()));
        if let (Some(s), Some(e)) = (start, end) {
            let mut ci = serde_json::Map::new();
            ci.insert("start_pos".to_string(), Value::Number(serde_json::Number::from(s as u64)));
            ci.insert("end_pos".to_string(), Value::Number(serde_json::Number::from(e as u64)));
            obj.insert("char_interval".to_string(), Value::Object(ci));
        }
        if let Some(attrs) = parent_attrs {
            if let Some(ps) = attrs.get("parent_step_id") {
                obj.insert("parent_step_id".to_string(), ps.clone());
            }
            if let Some(ps) = attrs.get("parent_start") {
                obj.insert("parent_start".to_string(), ps.clone());
            }
            if let Some(pe) = attrs.get("parent_end") {
                obj.insert("parent_end".to_string(), pe.clone());
            }
        }
        items.push(Value::Object(obj));
    }

    let mut items: Vec<Value> = Vec::new();

    // Map step id to name
    let mut step_id_to_name: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for s in &pipeline_result.config.steps {
        step_id_to_name.insert(s.id.as_str(), s.name.as_str());
    }

    for step_res in &pipeline_result.step_results {
        let step_name = step_id_to_name.get(step_res.step_id.as_str()).copied().unwrap_or("");
        for e in &step_res.extractions {
            // Determine absolute positions; fall back to exact match
            let (mut start, mut end) = (e.char_interval.as_ref().and_then(|ci| ci.start_pos), e.char_interval.as_ref().and_then(|ci| ci.end_pos));
            if start.is_none() || end.is_none() {
                if let Some(found) = original_text.find(&e.extraction_text) {
                    start = Some(found);
                    end = Some(found + e.extraction_text.len());
                }
            }

            // Push the main extraction
            push_item(
                &mut items,
                &e.extraction_class,
                &e.extraction_text,
                &step_res.step_id,
                step_name,
                start,
                end,
                e.attributes.as_ref(),
            );

            // Optionally expand nested JSON inside extraction_text
            if expand_nested_json {
                if let Ok(json_val) = serde_json::from_str::<Value>(&e.extraction_text) {
                    // Depth-first walk and collect leaf strings
                    fn collect(prefix: &str, val: &Value, out: &mut Vec<(String, String)>) {
                        match val {
                            Value::String(s) => out.push((prefix.to_string(), s.clone())),
                            Value::Object(map) => {
                                for (k, v) in map {
                                    let p = if prefix.is_empty() { k.clone() } else { format!("{}:{}", prefix, k) };
                                    collect(&p, v, out);
                                }
                            }
                            Value::Array(arr) => {
                                for (i, v) in arr.iter().enumerate() {
                                    let p = if prefix.is_empty() { format!("[{}]", i) } else { format!("{}:[{}]", prefix, i) };
                                    collect(&p, v, out);
                                }
                            }
                            _ => {}
                        }
                    }

                    let mut leafs: Vec<(String, String)> = Vec::new();
                    collect(&e.extraction_class, &json_val, &mut leafs);

                    for (cls, s) in leafs {
                        if s.is_empty() { continue; }
                        let (mut ls, mut le) = (None, None);
                        if let Some(found) = original_text.find(&s) {
                            ls = Some(found);
                            le = Some(found + s.len());
                        }
                        push_item(
                            &mut items,
                            &cls,
                            &s,
                            &step_res.step_id,
                            step_name,
                            ls,
                            le,
                            e.attributes.as_ref(),
                        );
                    }
                }
            }
        }
    }

    let mut root = serde_json::Map::new();
    root.insert("extractions".to_string(), Value::Array(items));
    let mut meta = serde_json::Map::new();
    meta.insert("steps".to_string(), Value::Number(serde_json::Number::from(pipeline_result.config.steps.len() as u64)));
    meta.insert("total_time_ms".to_string(), Value::Number(serde_json::Number::from(pipeline_result.total_time_ms)));
    meta.insert("expand_nested_json".to_string(), Value::Bool(expand_nested_json));
    root.insert("metadata".to_string(), Value::Object(meta));

    Ok(serde_json::to_string_pretty(&Value::Object(root))?)
}

#[derive(Debug, Clone)]
struct LayeredSpan {
    start: usize,
    end: usize,
    class_name: String,
    text: String,
    step_index: usize,
    parent_step_id: Option<String>,
    parent_class: Option<String>,
    parent_text: Option<String>,
}

fn build_layered_spans(pipeline_result: &PipelineResult, original_text: &str, expand_nested_json: bool) -> Vec<LayeredSpan> {
    // Map step_id to step index for stable coloring
    let mut step_id_to_index: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, s) in pipeline_result.config.steps.iter().enumerate() {
        step_id_to_index.insert(s.id.as_str(), i);
    }

    let mut spans = Vec::new();
    for step_res in &pipeline_result.step_results {
        let step_index = *step_id_to_index.get(step_res.step_id.as_str()).unwrap_or(&0);
        for e in &step_res.extractions {
            let mut added = false;
            if let Some(interval) = &e.char_interval {
                if let (Some(start), Some(end)) = (interval.start_pos, interval.end_pos) {
                    if start < end && end <= original_text.len() {
                        spans.push(LayeredSpan {
                            start,
                            end,
                            class_name: e.extraction_class.clone(),
                            text: e.extraction_text.clone(),
                            step_index,
                            parent_step_id: e.attributes.as_ref().and_then(|m| m.get("parent_step_id")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                            parent_class: e.attributes.as_ref().and_then(|m| m.get("parent_class")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                            parent_text: e.attributes.as_ref().and_then(|m| m.get("parent_text")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                        });
                        added = true;
                    }
                }
            }
            if !added {
                // Attempt exact match search in original text
                if let Some(found) = original_text.find(&e.extraction_text) {
                    let start = found;
                    let end = start + e.extraction_text.len();
                    if end <= original_text.len() {
                        spans.push(LayeredSpan {
                            start,
                            end,
                            class_name: e.extraction_class.clone(),
                            text: e.extraction_text.clone(),
                            step_index,
                            parent_step_id: e.attributes.as_ref().and_then(|m| m.get("parent_step_id")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                            parent_class: e.attributes.as_ref().and_then(|m| m.get("parent_class")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                            parent_text: e.attributes.as_ref().and_then(|m| m.get("parent_text")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                        });
                    }
                }
            }

            // Optional nested JSON expansion: create child spans for string values found in the original text
            if expand_nested_json {
                if let Ok(json_val) = serde_json::from_str::<Value>(&e.extraction_text) {
                    // Collect (class_name, text) pairs
                    fn collect_strings(prefix: &str, val: &Value, out: &mut Vec<(String, String)>) {
                        match val {
                            Value::String(s) => {
                                out.push((prefix.to_string(), s.clone()));
                            }
                            Value::Object(map) => {
                                for (k, v) in map {
                                    let new_prefix = if prefix.is_empty() { k.clone() } else { format!("{}:{}", prefix, k) };
                                    collect_strings(&new_prefix, v, out);
                                }
                            }
                            Value::Array(arr) => {
                                for (i, v) in arr.iter().enumerate() {
                                    let new_prefix = if prefix.is_empty() { format!("[{}]", i) } else { format!("{}:[{}]", prefix, i) };
                                    collect_strings(&new_prefix, v, out);
                                }
                            }
                            _ => {}
                        }
                    }

                    let mut pairs: Vec<(String, String)> = Vec::new();
                    collect_strings(&e.extraction_class, &json_val, &mut pairs);

                    let parent_step_id = e.attributes.as_ref().and_then(|m| m.get("parent_step_id")).and_then(|v| v.as_str()).map(|s| s.to_string());

                    for (class_name, s) in pairs {
                        if !s.is_empty() {
                            if let Some(found) = original_text.find(&s) {
                                let start = found;
                                let end = start + s.len();
                                if end <= original_text.len() {
                                    spans.push(LayeredSpan {
                                        start,
                                        end,
                                        class_name: class_name.clone(),
                                        text: s.clone(),
                                        step_index,
                                        parent_step_id: parent_step_id.clone(),
                                        parent_class: e.attributes.as_ref().and_then(|m| m.get("parent_class")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                                        parent_text: e.attributes.as_ref().and_then(|m| m.get("parent_text")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    spans
}

fn build_legend_html(pipeline_result: &PipelineResult) -> String {
    let mut step_id_to_index: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, s) in pipeline_result.config.steps.iter().enumerate() {
        step_id_to_index.insert(s.id.as_str(), i);
    }
    let mut items = String::new();
    for step in &pipeline_result.config.steps {
        let idx = *step_id_to_index.get(step.id.as_str()).unwrap_or(&0);
        items.push_str(&format!(r#"<span class="legend-item"><span class="badge step-{}"></span>Step {}: {}</span>"#, idx, idx + 1, html_escape(&step.name)));
    }
    format!(r#"<div class="legend">{}</div>"#, items)
}

fn build_extractions_list_html(spans: &[LayeredSpan]) -> String {
    let mut grouped: std::collections::BTreeMap<usize, Vec<&LayeredSpan>> = std::collections::BTreeMap::new();
    for s in spans {
        grouped.entry(s.step_index).or_default().push(s);
    }
    let mut html = String::new();
    for (step_idx, list) in grouped {
        html.push_str(&format!(r#"<h3>Step {}</h3>"#, step_idx + 1));
        html.push_str("<ul>");
        for s in list {
            let parent_info = match (&s.parent_class, &s.parent_text) {
                (Some(pc), Some(pt)) if !pc.is_empty() && !pt.is_empty() => format!(" (parent: [{}] {})", html_escape(pc), html_escape(pt)),
                _ => String::new(),
            };
            html.push_str(&format!(r#"<li><span class=\"step-{} extraction-highlight\">[{}] {}{}</span></li>"#, step_idx, html_escape(&s.class_name), html_escape(&s.text), parent_info));
        }
        html.push_str("</ul>");
    }
    html
}

/// Build HTML of text with layered spans. Currently uses non-overlapping simplification.
fn highlight_text_html_with_layers(
    text: &str,
    spans: &[LayeredSpan],
    allow_overlaps: bool,
) -> LangExtractResult<String> {
    if !allow_overlaps {
        let mut intervals: Vec<(usize, usize, usize)> = spans
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.start < s.end && s.end <= text.len() { Some((s.start, s.end, i)) } else { None })
            .collect();
        intervals.sort_by_key(|(start, end, _)| (*start, *end));

        let mut result = String::new();
        let mut last_pos = 0usize;
        let mut last_end = 0usize;
        for (start, end, idx) in intervals {
            if start < last_end { continue; }
            let safe_start = find_char_boundary(text, start);
            let safe_end = find_char_boundary(text, end);
            if safe_start > last_pos {
                result.push_str(&html_escape(&text[last_pos..safe_start]));
            }
            if safe_start < safe_end {
                let s = &spans[idx];
                let seg = &text[safe_start..safe_end];
                result.push_str(&format!(
                    r#"<span class="extraction-highlight step-{}" data-class="{}" data-text="{}" data-parent-class="{}">{}</span>"#,
                    s.step_index,
                    html_escape(&s.class_name),
                    html_escape(&s.text),
                    html_escape(s.parent_class.as_deref().unwrap_or("")),
                    html_escape(seg)
                ));
                last_pos = safe_end;
                last_end = safe_end;
            }
        }
        if last_pos < text.len() {
            result.push_str(&html_escape(&text[last_pos..]));
        }
        return Ok(result);
    }

    let mut events: Vec<(usize, bool, usize)> = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        if s.start < s.end && s.end <= text.len() {
            events.push((s.start, true, i));
            events.push((s.end, false, i));
        }
    }
    events.sort_by_key(|(pos, is_start, idx)| (*pos, !*is_start, spans[*idx].step_index));

    let mut result = String::new();
    let mut cursor = 0usize;
    let mut open: Vec<usize> = Vec::new();

    let mut push_plain = |from: usize, to: usize, out: &mut String| {
        if to > from {
            out.push_str(&html_escape(&text[from..to]));
        }
    };

    for (pos, is_start, idx) in events {
        let safe_pos = find_char_boundary(text, pos);
        if is_start {
            push_plain(cursor, safe_pos, &mut result);
            let s = &spans[idx];
            result.push_str(&format!(
                r#"<span class="extraction-highlight step-{}" data-class="{}" data-text="{}" data-parent-class="{}">{}</span>"#,
                s.step_index,
                html_escape(&s.class_name),
                html_escape(&s.text),
                html_escape(s.parent_class.as_deref().unwrap_or("")),
                html_escape(&text[cursor..safe_pos])
            ));
            open.push(idx);
            cursor = safe_pos;
        } else {
            push_plain(cursor, safe_pos, &mut result);
            cursor = safe_pos;
            if let Some(pos_in_open) = open.iter().rposition(|&j| j == idx) {
                for _ in pos_in_open..open.len() {
                    result.push_str("</span>");
                }
                open.remove(pos_in_open);
                for j in open.iter().copied() {
                    let s = &spans[j];
                    result.push_str(&format!(
                        r#"<span class="extraction-highlight step-{}" data-class="{}" data-text="{}" data-parent-class="{}">{}</span>"#,
                        s.step_index,
                        html_escape(&s.class_name),
                        html_escape(&s.text),
                        html_escape(s.parent_class.as_deref().unwrap_or("")),
                        html_escape(&text[cursor..safe_pos])
                    ));
                }
            }
        }
    }
    push_plain(cursor, text.len(), &mut result);
    for _ in 0..open.len() { result.push_str("</span>"); }
    Ok(result)
}

/// Visualize an annotated document (legacy function for backward compatibility)
pub fn visualize(
    annotated_document: &AnnotatedDocument,
    show_char_intervals: bool,
) -> LangExtractResult<String> {
    visualize_text(annotated_document, show_char_intervals)
}

/// Export as simple text format (original implementation)
fn visualize_text(
    annotated_document: &AnnotatedDocument,
    show_char_intervals: bool,
) -> LangExtractResult<String> {
    let mut result = String::new();
    
    result.push_str("EXTRACTION VISUALIZATION\n");
    result.push_str("=" .repeat(50).as_str());
    result.push('\n');
    
    // Show document text
    let text = annotated_document.text.as_deref().unwrap_or("No text");
    result.push_str(&format!("Document Text ({} chars):\n", text.len()));
    result.push_str(&format!("   {}\n\n", text));
    
    // Show extractions
    if let Some(extractions) = &annotated_document.extractions {
        result.push_str(&format!("Found {} Extractions:\n", extractions.len()));
        result.push_str("-".repeat(30).as_str());
        result.push('\n');
        
        for (i, extraction) in extractions.iter().enumerate() {
            result.push_str(&format!("{}. [{}] {}\n", 
                i + 1, 
                extraction.extraction_class, 
                extraction.extraction_text
            ));
            
            if show_char_intervals {
                if let Some(interval) = &extraction.char_interval {
                    result.push_str(&format!("   Position: {:?}\n", interval));
                }
            }
            
            if let Some(description) = &extraction.description {
                result.push_str(&format!("   Description: {}\n", description));
            }
            
            result.push('\n');
        }
    } else {
        result.push_str("No extractions found\n");
    }
    
    // Show statistics
    result.push_str("Statistics:\n");
    result.push_str("-".repeat(15).as_str());
    result.push('\n');
    result.push_str(&format!("   Document ID: {}\n", 
        annotated_document.document_id.as_deref().unwrap_or("None")));
    result.push_str(&format!("   Text Length: {} characters\n", text.len()));
    result.push_str(&format!("   Total Extractions: {}\n", annotated_document.extraction_count()));
    
    if let Some(extractions) = &annotated_document.extractions {
        // Count unique extraction classes
        let mut class_counts = std::collections::HashMap::new();
        for extraction in extractions {
            *class_counts.entry(&extraction.extraction_class).or_insert(0) += 1;
        }
        
        result.push_str("   Extraction Classes:\n");
        for (class, count) in class_counts {
            result.push_str(&format!("     • {}: {} instance(s)\n", class, count));
        }
    }
    
    Ok(result)
}

/// Export as rich HTML with highlighting and interactivity
fn export_html(
    annotated_document: &AnnotatedDocument,
    config: &ExportConfig,
) -> LangExtractResult<String> {
    let title = config.title.as_deref().unwrap_or("LangExtract Results");
    let text = annotated_document.text.as_deref().unwrap_or("No text");
    
    let mut html = String::new();
    
    // HTML Header
    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8fafc;
            color: #334155;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #1e293b;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .document-text {{
            background: #f1f5f9;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Monaco', 'Menlo', monospace;
            line-height: 1.6;
            white-space: pre-wrap;
            position: relative;
            margin-bottom: 20px;
        }}
        .extraction-highlight {{
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 3px;
            padding: 1px 2px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .extraction-highlight:hover {{
            background: rgba(59, 130, 246, 0.3);
            transform: translateY(-1px);
        }}
        .extractions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .extraction-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.2s ease;
        }}
        .extraction-card:hover {{
            border-color: #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        }}
        .extraction-class {{
            background: #3b82f6;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 8px;
        }}
        .extraction-text {{
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 8px;
        }}
        .extraction-meta {{
            font-size: 0.9em;
            color: #64748b;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .class-counts {{
            background: #f1f5f9;
            border-radius: 8px;
            padding: 20px;
        }}
        .class-count-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .class-count-item:last-child {{
            border-bottom: none;
        }}
        .class-badge {{
            background: #10b981;
            color: white;
            padding: 2px 6px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        {}
    </style>
</head>
<body>
"#, title, config.custom_css.as_deref().unwrap_or("")));

    // Header
    html.push_str(&format!(r#"    <div class="container">
        <div class="header">
            <h1>{}</h1>
        </div>
        <div class="content">
"#, title));

    // Document text section (with highlighting if enabled)
    if config.include_text {
        html.push_str(r#"            <div class="section">
                <h2>Document Text</h2>
                <div class="document-text">"#);
        
        if config.highlight_extractions {
            html.push_str(&highlight_text_html(text, annotated_document)?);
        } else {
            html.push_str(&html_escape(text));
        }
        
        html.push_str("</div>\n            </div>\n");
    }

    // Extractions section
    if let Some(extractions) = &annotated_document.extractions {
        html.push_str(&format!(r#"            <div class="section">
                <h2>Extractions ({} found)</h2>
                <div class="extractions-grid">
"#, extractions.len()));

        for extraction in extractions {
            html.push_str(&format!(r#"                    <div class="extraction-card">
                        <div class="extraction-class">{}</div>
                        <div class="extraction-text">{}</div>
"#, html_escape(&extraction.extraction_class), html_escape(&extraction.extraction_text)));

            if config.show_char_intervals {
                if let Some(interval) = &extraction.char_interval {
                    html.push_str(&format!(r#"                        <div class="extraction-meta">Position: {}-{}</div>
"#, interval.start_pos.unwrap_or(0), interval.end_pos.unwrap_or(0)));
                }
            }

            if let Some(description) = &extraction.description {
                html.push_str(&format!(r#"                        <div class="extraction-meta">Description: {}</div>
"#, html_escape(description)));
            }

            html.push_str("                    </div>\n");
        }
        
        html.push_str("                </div>\n            </div>\n");
    }

    // Statistics section
    if config.include_statistics {
        html.push_str(r#"            <div class="section">
                <h2>Statistics</h2>
                <div class="stats-grid">
"#);

        let extraction_count = annotated_document.extraction_count();
        html.push_str(&format!(r#"                    <div class="stat-card">
                        <div class="stat-number">{}</div>
                        <div class="stat-label">Total Extractions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{}</div>
                        <div class="stat-label">Characters</div>
                    </div>
"#, extraction_count, text.len()));

        if let Some(extractions) = &annotated_document.extractions {
            let class_counts = count_extraction_classes(extractions);
            html.push_str(&format!(r#"                    <div class="stat-card">
                        <div class="stat-number">{}</div>
                        <div class="stat-label">Unique Classes</div>
                    </div>
"#, class_counts.len()));

            html.push_str("                </div>\n");
            
            // Class breakdown
            html.push_str(r#"                <h3>Extraction Classes</h3>
                <div class="class-counts">
"#);
            
            for (class, count) in class_counts {
                html.push_str(&format!(r#"                    <div class="class-count-item">
                        <span>{}</span>
                        <span class="class-badge">{}</span>
                    </div>
"#, html_escape(class), count));
            }
            
            html.push_str("                </div>\n");
        } else {
            html.push_str("                </div>\n");
        }
        
        html.push_str("            </div>\n");
    }

    // Footer
    html.push_str(r#"        </div>
    </div>
    
    <script>
        // Add interactivity for extraction highlights
        document.querySelectorAll('.extraction-highlight').forEach(element => {
            element.addEventListener('click', function() {
                const className = this.getAttribute('data-class');
                const text = this.getAttribute('data-text');
                alert(`Extraction: ${className}\nText: ${text}`);
            });
        });
    </script>
</body>
</html>"#);

    Ok(html)
}

/// Helper function to escape HTML characters
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Helper function to find the nearest valid UTF-8 character boundary
fn find_char_boundary(text: &str, mut index: usize) -> usize {
    // Clamp to text length first
    if index >= text.len() {
        return text.len();
    }
    
    // If we're already at a character boundary, return as-is
    if text.is_char_boundary(index) {
        return index;
    }
    
    // Search backwards for the nearest character boundary
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    
    index
}

/// Helper function to highlight extractions in text
fn highlight_text_html(text: &str, annotated_document: &AnnotatedDocument) -> LangExtractResult<String> {
    if let Some(extractions) = &annotated_document.extractions {
        // Collect all valid intervals with their extraction info
        let mut intervals: Vec<(usize, usize, &Extraction)> = Vec::new();
        
        for extraction in extractions {
            if let Some(interval) = &extraction.char_interval {
                if let (Some(start), Some(end)) = (interval.start_pos, interval.end_pos) {
                    if start < end && end <= text.len() {
                        intervals.push((start, end, extraction));
                    }
                }
            }
        }
        
        // Sort by start position
        intervals.sort_by_key(|(start, _, _)| *start);
        
        // Remove overlapping intervals - keep the first one when intervals overlap
        let mut filtered_intervals = Vec::new();
        let mut last_end = 0;
        
        for (start, end, extraction) in intervals {
            if start >= last_end {
                filtered_intervals.push((start, end, extraction));
                last_end = end;
            } else {
                // Skip overlapping interval, but log it for debugging
                log::debug!("Skipping overlapping extraction: '{}' at {}-{} (overlaps with previous ending at {})", 
                    extraction.extraction_text, start, end, last_end);
            }
        }
        
        // Now build the HTML with non-overlapping intervals
        let mut result = String::new();
        let mut last_pos = 0;
        
        for (start, end, extraction) in filtered_intervals {
            // Ensure we're at valid UTF-8 boundaries
            let safe_start = find_char_boundary(text, start);
            let safe_end = find_char_boundary(text, end);
            
            // Add text before this extraction
            if safe_start > last_pos {
                let safe_last_pos = find_char_boundary(text, last_pos);
                if safe_last_pos < safe_start {
                    result.push_str(&html_escape(&text[safe_last_pos..safe_start]));
                }
            }
            
            // Add the highlighted extraction (only if we have valid boundaries)
            if safe_start < safe_end && safe_end <= text.len() {
                let actual_text = &text[safe_start..safe_end];
                result.push_str(&format!(
                    r#"<span class="extraction-highlight" data-class="{}" data-text="{}">{}</span>"#,
                    html_escape(&extraction.extraction_class),
                    html_escape(&extraction.extraction_text),
                    html_escape(actual_text)
                ));
                last_pos = safe_end;
            } else {
                // Skip invalid boundaries but log for debugging
                log::debug!("Skipping extraction with invalid UTF-8 boundaries: '{}' at {}-{}", 
                    extraction.extraction_text, start, end);
            }
        }
        
        // Add remaining text
        if last_pos < text.len() {
            let safe_last_pos = find_char_boundary(text, last_pos);
            if safe_last_pos < text.len() {
                result.push_str(&html_escape(&text[safe_last_pos..]));
            }
        }
        
        Ok(result)
    } else {
        Ok(html_escape(text))
    }
}

/// Helper function to count extraction classes
fn count_extraction_classes(extractions: &[crate::data::Extraction]) -> HashMap<&str, usize> {
    let mut class_counts = HashMap::new();
    for extraction in extractions {
        *class_counts.entry(extraction.extraction_class.as_str()).or_insert(0) += 1;
    }
    class_counts
}

/// Export as structured markdown with extraction summaries
fn export_markdown(
    annotated_document: &AnnotatedDocument,
    config: &ExportConfig,
) -> LangExtractResult<String> {
    let title = config.title.as_deref().unwrap_or("LangExtract Results");
    let text = annotated_document.text.as_deref().unwrap_or("No text");
    
    let mut md = String::new();
    
    // Title
    md.push_str(&format!("# {}\n\n", title));
    
    // Document text section
    if config.include_text {
        md.push_str("## Document Text\n\n");
        
        if config.highlight_extractions {
            md.push_str(&highlight_text_markdown(text, annotated_document)?);
        } else {
            md.push_str(&format!("```\n{}\n```\n", text));
        }
        
        md.push_str("\n");
    }
    
    // Extractions section
    if let Some(extractions) = &annotated_document.extractions {
        md.push_str(&format!("## Extractions ({} found)\n\n", extractions.len()));
        
        for (i, extraction) in extractions.iter().enumerate() {
            md.push_str(&format!("### {}. {}\n\n", i + 1, extraction.extraction_class));
            md.push_str(&format!("**Text:** {}\n\n", extraction.extraction_text));
            
            if config.show_char_intervals {
                if let Some(interval) = &extraction.char_interval {
                    md.push_str(&format!("**Position:** {}-{}\n\n", interval.start_pos.unwrap_or(0), interval.end_pos.unwrap_or(0)));
                }
            }
            
            if let Some(description) = &extraction.description {
                md.push_str(&format!("**Description:** {}\n\n", description));
            }
        }
    }
    
    // Statistics section
    if config.include_statistics {
        md.push_str("## Statistics\n\n");
        
        let extraction_count = annotated_document.extraction_count();
        md.push_str(&format!("- **Total Extractions:** {}\n", extraction_count));
        md.push_str(&format!("- **Text Length:** {} characters\n", text.len()));
        
        if let Some(extractions) = &annotated_document.extractions {
            let class_counts = count_extraction_classes(extractions);
            md.push_str(&format!("- **Unique Classes:** {}\n\n", class_counts.len()));
            
            md.push_str("### Extraction Classes\n\n");
            md.push_str("| Class | Count |\n");
            md.push_str("|-------|-------|\n");
            
            for (class, count) in class_counts {
                md.push_str(&format!("| {} | {} |\n", class, count));
            }
        }
        
        md.push_str("\n");
    }
    
    Ok(md)
}

/// Helper function to highlight extractions in markdown
fn highlight_text_markdown(text: &str, annotated_document: &AnnotatedDocument) -> LangExtractResult<String> {
    if let Some(extractions) = &annotated_document.extractions {
        let mut result = String::new();
        let mut last_pos = 0;
        
        // Sort extractions by start position
        let mut sorted_extractions: Vec<_> = extractions.iter().collect();
        sorted_extractions.sort_by_key(|e| {
            e.char_interval.as_ref().and_then(|i| i.start_pos).unwrap_or(usize::MAX)
        });
        
        result.push_str("```\n");
        
        for extraction in sorted_extractions {
            if let Some(interval) = &extraction.char_interval {
                // Add text before the extraction
                if interval.start_pos.unwrap_or(0) > last_pos && interval.start_pos.unwrap_or(0) <= text.len() {
                    result.push_str(&text[last_pos..interval.start_pos.unwrap_or(0)]);
                }
                
                // Add highlighted extraction with markdown bold
                if interval.end_pos.unwrap_or(0) <= text.len() && interval.start_pos.unwrap_or(0) < interval.end_pos.unwrap_or(0) {
                    let extraction_text = &text[interval.start_pos.unwrap_or(0)..interval.end_pos.unwrap_or(0)];
                    result.push_str(&format!("**{}**", extraction_text));
                    last_pos = interval.end_pos.unwrap_or(0);
                }
            }
        }
        
        // Add remaining text
        if last_pos < text.len() {
            result.push_str(&text[last_pos..]);
        }
        
        result.push_str("\n```\n");
        Ok(result)
    } else {
        Ok(format!("```\n{}\n```\n", text))
    }
}

/// Export as JSON for analysis
fn export_json(
    annotated_document: &AnnotatedDocument,
    config: &ExportConfig,
) -> LangExtractResult<String> {
    let mut json_data = json!({
        "document_id": annotated_document.document_id,
        "export_config": {
            "format": "json",
            "show_char_intervals": config.show_char_intervals,
            "include_text": config.include_text,
            "include_statistics": config.include_statistics,
            "title": config.title
        }
    });
    
    // Add text if requested
    if config.include_text {
        json_data["text"] = json!(annotated_document.text);
    }
    
    // Add extractions
    if let Some(extractions) = &annotated_document.extractions {
        let extractions_json: Vec<Value> = extractions.iter().map(|extraction| {
            let mut ext_json = json!({
                "extraction_class": extraction.extraction_class,
                "extraction_text": extraction.extraction_text,
                "description": extraction.description
            });
            
            if config.show_char_intervals {
                if let Some(interval) = &extraction.char_interval {
                    ext_json["char_interval"] = json!({
                        "start_char": interval.start_pos.unwrap_or(0),
                        "end_char": interval.end_pos.unwrap_or(0),
                        "alignment_status": extraction.alignment_status.as_ref().map(|s| format!("{:?}", s)).unwrap_or_else(|| "None".to_string())
                    });
                }
            }
            
            if let Some(group_index) = extraction.group_index {
                ext_json["group_index"] = json!(group_index);
            }
            
            ext_json
        }).collect();
        
        json_data["extractions"] = json!(extractions_json);
    }
    
    // Add statistics if requested
    if config.include_statistics {
        let text = annotated_document.text.as_deref().unwrap_or("");
        let mut stats = json!({
            "total_extractions": annotated_document.extraction_count(),
            "text_length": text.len()
        });
        
        if let Some(extractions) = &annotated_document.extractions {
            let class_counts = count_extraction_classes(extractions);
            stats["unique_classes"] = json!(class_counts.len());
            stats["extraction_classes"] = json!(class_counts);
        }
        
        json_data["statistics"] = stats;
    }
    
    Ok(serde_json::to_string_pretty(&json_data)?)
}

/// Export as CSV for spreadsheet analysis
fn export_csv(
    annotated_document: &AnnotatedDocument,
    config: &ExportConfig,
) -> LangExtractResult<String> {
    let mut csv = String::new();
    
    // CSV Header
    if config.show_char_intervals {
        csv.push_str("extraction_class,extraction_text,description,start_char,end_char,alignment_status,group_index\n");
    } else {
        csv.push_str("extraction_class,extraction_text,description,group_index\n");
    }
    
    // CSV Rows
    if let Some(extractions) = &annotated_document.extractions {
        for extraction in extractions {
            let class = csv_escape(&extraction.extraction_class);
            let text = csv_escape(&extraction.extraction_text);
            let description = extraction.description.as_ref().map(|d| csv_escape(d)).unwrap_or_else(|| "".to_string());
            let group_index = extraction.group_index.map(|i| i.to_string()).unwrap_or_else(|| "".to_string());
            
            if config.show_char_intervals {
                if let Some(interval) = &extraction.char_interval {
                    csv.push_str(&format!("{},{},{},{},{},{:?},{}\n",
                        class, text, description,
                        interval.start_pos.unwrap_or(0), interval.end_pos.unwrap_or(0),
                        extraction.alignment_status.as_ref().map(|s| format!("{:?}", s)).unwrap_or_else(|| "None".to_string()), group_index));
                } else {
                    csv.push_str(&format!("{},{},{},,,None,{}\n",
                        class, text, description, group_index));
                }
            } else {
                csv.push_str(&format!("{},{},{},{}\n",
                    class, text, description, group_index));
            }
        }
    }
    
    Ok(csv)
}

/// Helper function to escape CSV values
fn csv_escape(text: &str) -> String {
    if text.contains(',') || text.contains('"') || text.contains('\n') {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{AlignmentStatus, CharInterval, Extraction};
    use std::collections::HashMap;
    use crate::pipeline::{PipelineConfig, PipelineStep, StepResult, PipelineResult};
    use crate::ExtractConfig as LibExtractConfig;

    fn create_sample_document() -> AnnotatedDocument {
        let text = "John Smith works at TechCorp and earns $50,000.";
        let extractions = vec![
            Extraction {
                extraction_class: "person".to_string(),
                extraction_text: "John Smith".to_string(),
                char_interval: Some(CharInterval::new(Some(0), Some(10))),
                alignment_status: Some(AlignmentStatus::MatchExact),
                extraction_index: Some(0),
                group_index: Some(0),
                description: Some("Person name".to_string()),
                attributes: Some(HashMap::new()),
                token_interval: None,
            },
            Extraction {
                extraction_class: "company".to_string(),
                extraction_text: "TechCorp".to_string(),
                char_interval: Some(CharInterval::new(Some(20), Some(28))),
                alignment_status: Some(AlignmentStatus::MatchExact),
                extraction_index: Some(1),
                group_index: Some(0),
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            },
            Extraction {
                extraction_class: "salary".to_string(),
                extraction_text: "$50,000".to_string(),
                char_interval: Some(CharInterval::new(Some(39), Some(46))),
                alignment_status: Some(AlignmentStatus::MatchFuzzy),
                extraction_index: Some(2),
                group_index: Some(0),
                description: Some("Annual salary".to_string()),
                attributes: Some(HashMap::new()),
                token_interval: None,
            },
        ];

        AnnotatedDocument {
            document_id: Some("test_doc".to_string()),
            text: Some(text.to_string()),
            extractions: Some(extractions),
        }
    }

    #[test]
    fn test_text_export() {
        let document = create_sample_document();
        let config = ExportConfig {
            format: ExportFormat::Text,
            show_char_intervals: true,
            ..Default::default()
        };

        let result = export_document(&document, &config).unwrap();
        
        assert!(result.contains("EXTRACTION VISUALIZATION"));
        assert!(result.contains("John Smith"));
        assert!(result.contains("TechCorp"));
        assert!(result.contains("$50,000"));
        assert!(result.contains("Position:"));
        assert!(result.contains("Statistics:"));
    }

    #[test]
    fn test_html_export() {
        let document = create_sample_document();
        let config = ExportConfig {
            format: ExportFormat::Html,
            title: Some("Test HTML Export".to_string()),
            highlight_extractions: true,
            show_char_intervals: true,
            ..Default::default()
        };

        let result = export_document(&document, &config).unwrap();
        
        assert!(result.contains("<!DOCTYPE html>"));
        assert!(result.contains("<title>Test HTML Export</title>"));
        assert!(result.contains("extraction-highlight"));
        assert!(result.contains("John Smith"));
        assert!(result.contains("TechCorp"));
        assert!(result.contains("extraction-card"));
        assert!(result.contains("stats-grid"));
        assert!(result.contains("</html>"));
    }

    #[test]
    fn test_html_export_with_custom_css() {
        let document = create_sample_document();
        let custom_css = "body { background: red; }";
        let config = ExportConfig {
            format: ExportFormat::Html,
            custom_css: Some(custom_css.to_string()),
            ..Default::default()
        };

        let result = export_document(&document, &config).unwrap();
        
        assert!(result.contains(custom_css));
    }

    #[test]
    fn test_markdown_export() {
        let document = create_sample_document();
        let config = ExportConfig {
            format: ExportFormat::Markdown,
            title: Some("Test Markdown".to_string()),
            show_char_intervals: true,
            highlight_extractions: true,
            ..Default::default()
        };

        let result = export_document(&document, &config).unwrap();
        
        assert!(result.starts_with("# Test Markdown"));
        assert!(result.contains("## Document Text"));
        assert!(result.contains("## Extractions"));
        assert!(result.contains("### 1. person"));
        assert!(result.contains("**Text:** John Smith"));
        assert!(result.contains("**Position:** 0-10"));
        assert!(result.contains("| Class | Count |"));
        assert!(result.contains("| person | 1 |"));
    }

    #[test]
    fn test_json_export() {
        let document = create_sample_document();
        let config = ExportConfig {
            format: ExportFormat::Json,
            show_char_intervals: true,
            include_text: true,
            include_statistics: true,
            ..Default::default()
        };

        let result = export_document(&document, &config).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        
        assert_eq!(parsed["document_id"], "test_doc");
        assert!(parsed["text"].is_string());
        assert!(parsed["extractions"].is_array());
        assert!(parsed["statistics"].is_object());
        
        let extractions = parsed["extractions"].as_array().unwrap();
        assert_eq!(extractions.len(), 3);
        
        let first_extraction = &extractions[0];
        assert_eq!(first_extraction["extraction_class"], "person");
        assert_eq!(first_extraction["extraction_text"], "John Smith");
        assert!(first_extraction["char_interval"].is_object());
        
        let stats = &parsed["statistics"];
        assert_eq!(stats["total_extractions"], 3);
        assert_eq!(stats["unique_classes"], 3);
    }

    #[test]
    fn test_csv_export() {
        let document = create_sample_document();
        let config = ExportConfig {
            format: ExportFormat::Csv,
            show_char_intervals: true,
            ..Default::default()
        };

        let result = export_document(&document, &config).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        
        // Check header
        assert_eq!(lines[0], "extraction_class,extraction_text,description,start_char,end_char,alignment_status,group_index");
        
        // Check data rows
        assert_eq!(lines.len(), 4); // Header + 3 data rows
        assert!(lines[1].contains("person,John Smith"));
        assert!(lines[2].contains("company,TechCorp"));
        assert!(lines[3].contains("salary,\"$50,000\""));
        assert!(lines[1].contains("MatchExact"));
        assert!(lines[3].contains("MatchFuzzy"));
    }

    #[test]
    fn test_csv_export_without_intervals() {
        let document = create_sample_document();
        let config = ExportConfig {
            format: ExportFormat::Csv,
            show_char_intervals: false,
            ..Default::default()
        };

        let result = export_document(&document, &config).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        
        // Check header
        assert_eq!(lines[0], "extraction_class,extraction_text,description,group_index");
        
        // Should not contain position columns
        assert!(!result.contains("start_char"));
        assert!(!result.contains("end_char"));
    }

    #[test]
    fn test_csv_escape() {
        assert_eq!(csv_escape("simple"), "simple");
        assert_eq!(csv_escape("has,comma"), "\"has,comma\"");
        assert_eq!(csv_escape("has\"quote"), "\"has\"\"quote\"");
        assert_eq!(csv_escape("has\nnewline"), "\"has\nnewline\"");
        assert_eq!(csv_escape("has,comma\"and quote"), "\"has,comma\"\"and quote\"");
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("simple"), "simple");
        assert_eq!(html_escape("has<tag>"), "has&lt;tag&gt;");
        assert_eq!(html_escape("has\"quote"), "has&quot;quote");
        assert_eq!(html_escape("has'apostrophe"), "has&#x27;apostrophe");
        assert_eq!(html_escape("has&ampersand"), "has&amp;ampersand");
    }

    #[test]
    fn test_export_config_defaults() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::Text);
        assert!(!config.show_char_intervals);
        assert!(config.include_text);
        assert!(config.highlight_extractions);
        assert!(config.include_statistics);
        assert!(config.custom_css.is_none());
        assert!(config.title.is_none());
    }

    #[test]
    fn test_empty_document() {
        let document = AnnotatedDocument {
            document_id: Some("empty".to_string()),
            text: Some("".to_string()),
            extractions: None,
        };

        let config = ExportConfig::default();
        let result = export_document(&document, &config).unwrap();
        
        assert!(result.contains("No extractions found"));
    }

    #[test]
    fn test_document_without_text() {
        let document = AnnotatedDocument {
            document_id: Some("no_text".to_string()),
            text: None,
            extractions: None,
        };

        let config = ExportConfig::default();
        let result = export_document(&document, &config).unwrap();
        
        assert!(result.contains("No text"));
    }

    #[test]
    fn test_export_format_variants() {
        let document = create_sample_document();
        
        // Test all export formats don't panic
        for format in [ExportFormat::Text, ExportFormat::Html, ExportFormat::Markdown, ExportFormat::Json, ExportFormat::Csv] {
            let config = ExportConfig {
                format,
                ..Default::default()
            };
            let result = export_document(&document, &config);
            assert!(result.is_ok(), "Format {:?} failed", format);
        }
    }

    #[test]
    fn test_highlight_text_html() {
        let document = create_sample_document();
        let text = document.text.as_ref().unwrap();
        
        let result = highlight_text_html(text, &document).unwrap();
        
        assert!(result.contains("extraction-highlight"));
        assert!(result.contains("data-class=\"person\""));
        assert!(result.contains("data-text=\"John Smith\""));
        assert!(result.contains("John Smith"));
    }

    #[test]
    fn test_count_extraction_classes() {
        let extractions = vec![
            Extraction {
                extraction_class: "person".to_string(),
                extraction_text: "John".to_string(),
                char_interval: None,
                alignment_status: None,
                extraction_index: None,
                group_index: None,
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            },
            Extraction {
                extraction_class: "person".to_string(),
                extraction_text: "Jane".to_string(),
                char_interval: None,
                alignment_status: None,
                extraction_index: None,
                group_index: None,
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            },
            Extraction {
                extraction_class: "company".to_string(),
                extraction_text: "TechCorp".to_string(),
                char_interval: None,
                alignment_status: None,
                extraction_index: None,
                group_index: None,
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            },
        ];

        let counts = count_extraction_classes(&extractions);
        
        assert_eq!(counts.get("person"), Some(&2));
        assert_eq!(counts.get("company"), Some(&1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_export_pipeline_html_renders_layers() {
        // Original text
        let text = "The system shall process 100 transactions per second.";

        // Step definitions
        let steps = vec![
            PipelineStep {
                id: "s1".to_string(),
                name: "Extract Requirements".to_string(),
                description: "".to_string(),
                examples: vec![],
                prompt: "".to_string(),
                output_field: "requirements".to_string(),
                filter: None,
                depends_on: vec![],
            },
            PipelineStep {
                id: "s2".to_string(),
                name: "Extract Values".to_string(),
                description: "".to_string(),
                examples: vec![],
                prompt: "".to_string(),
                output_field: "values".to_string(),
                filter: None,
                depends_on: vec!["s1".to_string()],
            },
        ];

        let cfg = PipelineConfig {
            name: "Test".to_string(),
            description: "".to_string(),
            version: "0.0.0".to_string(),
            steps: steps.clone(),
            global_config: LibExtractConfig::default(),
            enable_parallel_execution: false,
        };

        // Compute positions
        let parent_start = 0usize;
        let parent_end = text.len();
        let hundred_idx = text.find("100").unwrap();
        let unit_idx = text.find("transactions per second").unwrap();

        let step1_res = StepResult {
            step_id: "s1".to_string(),
            step_name: "Extract Requirements".to_string(),
            extractions: vec![Extraction {
                extraction_class: "requirement".to_string(),
                extraction_text: text.to_string(),
                char_interval: Some(CharInterval::new(Some(parent_start), Some(parent_end))),
                alignment_status: Some(AlignmentStatus::MatchExact),
                extraction_index: Some(0),
                group_index: None,
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            }],
            processing_time_ms: 1,
            input_count: 1,
            success: true,
            error_message: None,
        };

        let step2_res = StepResult {
            step_id: "s2".to_string(),
            step_name: "Extract Values".to_string(),
            extractions: vec![
                Extraction {
                    extraction_class: "value".to_string(),
                    extraction_text: "100".to_string(),
                    char_interval: Some(CharInterval::new(Some(hundred_idx), Some(hundred_idx + 3))),
                    alignment_status: Some(AlignmentStatus::MatchExact),
                    extraction_index: Some(0),
                    group_index: None,
                    description: None,
                    attributes: Some(HashMap::new()),
                    token_interval: None,
                },
                Extraction {
                    extraction_class: "unit".to_string(),
                    extraction_text: "transactions per second".to_string(),
                    char_interval: Some(CharInterval::new(Some(unit_idx), Some(unit_idx + "transactions per second".len()))),
                    alignment_status: Some(AlignmentStatus::MatchExact),
                    extraction_index: Some(1),
                    group_index: None,
                    description: None,
                    attributes: Some(HashMap::new()),
                    token_interval: None,
                }
            ],
            processing_time_ms: 1,
            input_count: 1,
            success: true,
            error_message: None,
        };

        let pr = PipelineResult {
            config: cfg,
            step_results: vec![step1_res, step2_res],
            nested_output: serde_json::json!({}),
            total_time_ms: 2,
            success: true,
            error_message: None,
        };

        let config = ExportConfig { format: ExportFormat::Html, ..Default::default() };
        let html = export_pipeline_html(&pr, text, &config).unwrap();
        assert!(html.contains("step-0"), "Should render step-0 (parent)");
        assert!(html.contains("step-1"), "Should render step-1 (child)");
        assert!(html.contains("100"));
        assert!(html.contains("transactions per second"));
    }

    #[test]
    fn test_export_pipeline_html_exact_match_fallback() {
        let text = "System uptime must be 99.9% for availability.";

        let steps = vec![
            PipelineStep { id: "s1".to_string(), name: "Req".to_string(), description: "".to_string(), examples: vec![], prompt: "".to_string(), output_field: "req".to_string(), filter: None, depends_on: vec![] },
            PipelineStep { id: "s2".to_string(), name: "Vals".to_string(), description: "".to_string(), examples: vec![], prompt: "".to_string(), output_field: "vals".to_string(), filter: None, depends_on: vec!["s1".to_string()] },
        ];
        let cfg = PipelineConfig { name: "T".to_string(), description: "".to_string(), version: "0".to_string(), steps, global_config: LibExtractConfig::default(), enable_parallel_execution: false };

        let step1_res = StepResult {
            step_id: "s1".to_string(),
            step_name: "Req".to_string(),
            extractions: vec![Extraction {
                extraction_class: "requirement".to_string(),
                extraction_text: text.to_string(),
                char_interval: None, // Not strictly needed for this test
                alignment_status: None,
                extraction_index: None,
                group_index: None,
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            }],
            processing_time_ms: 1,
            input_count: 1,
            success: true,
            error_message: None,
        };

        let step2_res = StepResult {
            step_id: "s2".to_string(),
            step_name: "Vals".to_string(),
            extractions: vec![Extraction {
                extraction_class: "uptime".to_string(),
                extraction_text: "99.9%".to_string(),
                char_interval: None, // Force fallback
                alignment_status: None,
                extraction_index: None,
                group_index: None,
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            }],
            processing_time_ms: 1,
            input_count: 1,
            success: true,
            error_message: None,
        };

        let pr = PipelineResult { config: cfg, step_results: vec![step1_res, step2_res], nested_output: serde_json::json!({}), total_time_ms: 2, success: true, error_message: None };

        let config = ExportConfig { format: ExportFormat::Html, ..Default::default() };
        let html = export_pipeline_html(&pr, text, &config).unwrap();
        assert!(html.contains("99.9%"), "Fallback should highlight exact match in original text");
    }

    #[test]
    fn test_export_pipeline_html_overlap_rendering() {
        // Overlapping child spans inside a parent requirement
        let text = "The system shall support 10 users concurrently.";

        let steps = vec![
            PipelineStep { id: "s1".to_string(), name: "Req".to_string(), description: "".to_string(), examples: vec![], prompt: "".to_string(), output_field: "req".to_string(), filter: None, depends_on: vec![] },
            PipelineStep { id: "s2".to_string(), name: "Vals".to_string(), description: "".to_string(), examples: vec![], prompt: "".to_string(), output_field: "vals".to_string(), filter: None, depends_on: vec!["s1".to_string()] },
        ];
        let cfg = PipelineConfig { name: "T".to_string(), description: "".to_string(), version: "0".to_string(), steps, global_config: LibExtractConfig::default(), enable_parallel_execution: false };

        let parent_start = 0usize;
        let parent_end = text.len();
        let ten_idx = text.find("10").unwrap();
        let users_idx = text.find("10 users").unwrap();

        let step1_res = StepResult {
            step_id: "s1".to_string(),
            step_name: "Req".to_string(),
            extractions: vec![Extraction {
                extraction_class: "requirement".to_string(),
                extraction_text: text.to_string(),
                char_interval: Some(CharInterval::new(Some(parent_start), Some(parent_end))),
                alignment_status: None,
                extraction_index: None,
                group_index: None,
                description: None,
                attributes: Some(HashMap::new()),
                token_interval: None,
            }],
            processing_time_ms: 1,
            input_count: 1,
            success: true,
            error_message: None,
        };

        let step2_res = StepResult {
            step_id: "s2".to_string(),
            step_name: "Vals".to_string(),
            extractions: vec![
                Extraction {
                    extraction_class: "value".to_string(),
                    extraction_text: "10".to_string(),
                    char_interval: Some(CharInterval::new(Some(ten_idx), Some(ten_idx + 2))),
                    alignment_status: None,
                    extraction_index: None,
                    group_index: None,
                    description: None,
                    attributes: Some(HashMap::new()),
                    token_interval: None,
                },
                Extraction {
                    extraction_class: "phrase".to_string(),
                    extraction_text: "10 users".to_string(),
                    char_interval: Some(CharInterval::new(Some(users_idx), Some(users_idx + "10 users".len()))),
                    alignment_status: None,
                    extraction_index: None,
                    group_index: None,
                    description: None,
                    attributes: Some(HashMap::new()),
                    token_interval: None,
                },
            ],
            processing_time_ms: 1,
            input_count: 1,
            success: true,
            error_message: None,
        };

        let pr = PipelineResult { config: cfg, step_results: vec![step1_res, step2_res], nested_output: serde_json::json!({}), total_time_ms: 2, success: true, error_message: None };

        let mut config = ExportConfig { format: ExportFormat::Html, ..Default::default() };
        config.allow_overlapping_highlights = true;
        let html = export_pipeline_html(&pr, text, &config).unwrap();
        // Should include both occurrences
        assert!(html.contains("10"));
        assert!(html.contains("10 users"));
    }
}
