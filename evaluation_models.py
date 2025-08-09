#!/usr/bin/env python3
"""
Data models for OCR evaluation system
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
import json


class FieldImportance(str, Enum):
    """Field importance levels for evaluation weighting"""
    CRITICAL = "critical"  # Must be 100% accurate (invoice_number, total_amount)
    HIGH = "high"        # High importance (customer_id, invoice_date)
    MEDIUM = "medium"    # Medium importance (line_items, addresses)
    LOW = "low"         # Low importance (email metadata)


class GroundTruthValue(BaseModel):
    """Ground truth value for a specific field"""
    field_name: str
    value: Union[str, float, int, List[Dict], None]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    notes: Optional[str] = None
    annotated_by: Optional[str] = None
    annotated_at: datetime = Field(default_factory=datetime.now)
    importance: FieldImportance = FieldImportance.HIGH


class OCRResult(BaseModel):
    """Result from an OCR tool for comparison"""
    tool_name: str
    tool_version: Optional[str] = None
    extracted_values: Dict[str, Any]
    processing_time: Optional[float] = None
    cost: Optional[float] = None
    confidence_scores: Optional[Dict[str, float]] = None
    raw_output: Optional[Dict[str, Any]] = None
    processed_at: datetime = Field(default_factory=datetime.now)


class EvaluationDocument(BaseModel):
    """Document in evaluation dataset with ground truth and OCR results"""
    document_id: str
    file_path: str
    file_hash: str
    document_type: str = "waste_management_invoice"
    ground_truth: Dict[str, GroundTruthValue] = Field(default_factory=dict)
    ocr_results: Dict[str, OCRResult] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_ground_truth(self, field_name: str, value: Any, importance: FieldImportance = FieldImportance.HIGH, 
                        confidence: float = 1.0, notes: Optional[str] = None, annotated_by: Optional[str] = None):
        """Add ground truth value for a field"""
        self.ground_truth[field_name] = GroundTruthValue(
            field_name=field_name,
            value=value,
            confidence=confidence,
            notes=notes,
            annotated_by=annotated_by,
            importance=importance
        )
        self.updated_at = datetime.now()
    
    def add_ocr_result(self, tool_name: str, extracted_values: Dict[str, Any], 
                      tool_version: Optional[str] = None, processing_time: Optional[float] = None,
                      cost: Optional[float] = None, confidence_scores: Optional[Dict[str, float]] = None,
                      raw_output: Optional[Dict[str, Any]] = None):
        """Add OCR result from a tool"""
        self.ocr_results[tool_name] = OCRResult(
            tool_name=tool_name,
            tool_version=tool_version,
            extracted_values=extracted_values,
            processing_time=processing_time,
            cost=cost,
            confidence_scores=confidence_scores,
            raw_output=raw_output
        )
        self.updated_at = datetime.now()


class FieldAccuracy(BaseModel):
    """Accuracy metrics for a specific field"""
    field_name: str
    tool_name: str
    correct_predictions: int = 0
    total_predictions: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    accuracy: float = 0.0
    exact_match_rate: float = 0.0
    partial_match_rate: float = 0.0
    importance_weighted_score: float = 0.0


class EvaluationMetrics(BaseModel):
    """Overall evaluation metrics for OCR tools"""
    tool_name: str
    total_documents: int = 0
    overall_accuracy: float = 0.0
    field_accuracies: Dict[str, FieldAccuracy] = Field(default_factory=dict)
    importance_weighted_score: float = 0.0
    processing_stats: Dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=datetime.now)


class EvaluationDataset(BaseModel):
    """Complete evaluation dataset"""
    dataset_id: str
    name: str
    description: Optional[str] = None
    documents: Dict[str, EvaluationDocument] = Field(default_factory=dict)
    metrics: Dict[str, EvaluationMetrics] = Field(default_factory=dict)
    field_importance_weights: Dict[str, FieldImportance] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_document(self, document: EvaluationDocument):
        """Add a document to the dataset"""
        self.documents[document.document_id] = document
        self.updated_at = datetime.now()
    
    def get_field_importance_weight(self, importance: FieldImportance) -> float:
        """Get numeric weight for field importance"""
        weights = {
            FieldImportance.CRITICAL: 4.0,
            FieldImportance.HIGH: 3.0,
            FieldImportance.MEDIUM: 2.0,
            FieldImportance.LOW: 1.0
        }
        return weights.get(importance, 2.0)
    
    def calculate_metrics(self):
        """Calculate evaluation metrics for all OCR tools"""
        # Get all unique OCR tools
        all_tools = set()
        for doc in self.documents.values():
            all_tools.update(doc.ocr_results.keys())
        
        # Calculate metrics for each tool
        for tool_name in all_tools:
            metrics = EvaluationMetrics(tool_name=tool_name)
            field_stats = {}
            
            for doc in self.documents.values():
                if tool_name not in doc.ocr_results:
                    continue
                    
                metrics.total_documents += 1
                ocr_result = doc.ocr_results[tool_name]
                
                # Compare each field with ground truth
                for field_name, ground_truth in doc.ground_truth.items():
                    if field_name not in field_stats:
                        field_stats[field_name] = FieldAccuracy(
                            field_name=field_name,
                            tool_name=tool_name
                        )
                    
                    field_acc = field_stats[field_name]
                    field_acc.total_predictions += 1
                    
                    # Get predicted value
                    predicted_value = ocr_result.extracted_values.get(field_name)
                    actual_value = ground_truth.value
                    
                    # Calculate accuracy based on field type
                    is_correct, is_exact, is_partial = self._compare_values(predicted_value, actual_value)
                    
                    if is_correct:
                        field_acc.correct_predictions += 1
                    if is_exact:
                        field_acc.exact_matches += 1
                    if is_partial:
                        field_acc.partial_matches += 1
            
            # Calculate final metrics
            for field_name, field_acc in field_stats.items():
                if field_acc.total_predictions > 0:
                    field_acc.accuracy = field_acc.correct_predictions / field_acc.total_predictions
                    field_acc.exact_match_rate = field_acc.exact_matches / field_acc.total_predictions
                    field_acc.partial_match_rate = field_acc.partial_matches / field_acc.total_predictions
                    
                    # Get importance weight
                    importance = self.field_importance_weights.get(field_name, FieldImportance.MEDIUM)
                    weight = self.get_field_importance_weight(importance)
                    field_acc.importance_weighted_score = field_acc.accuracy * weight
                
                metrics.field_accuracies[field_name] = field_acc
            
            # Calculate overall metrics
            if field_stats:
                total_weighted_score = sum(acc.importance_weighted_score for acc in field_stats.values())
                total_weight = sum(self.get_field_importance_weight(
                    self.field_importance_weights.get(fname, FieldImportance.MEDIUM)
                ) for fname in field_stats.keys())
                
                metrics.importance_weighted_score = total_weighted_score / total_weight if total_weight > 0 else 0
                metrics.overall_accuracy = sum(acc.accuracy for acc in field_stats.values()) / len(field_stats)
            
            self.metrics[tool_name] = metrics
        
        self.updated_at = datetime.now()
    
    def _compare_values(self, predicted, actual) -> tuple[bool, bool, bool]:
        """Compare predicted vs actual values, return (is_correct, is_exact_match, is_partial_match)"""
        if predicted is None and actual is None:
            return True, True, False
        
        if predicted is None or actual is None:
            return False, False, False
        
        # Convert to strings for comparison
        pred_str = str(predicted).strip().lower()
        actual_str = str(actual).strip().lower()
        
        # Exact match
        if pred_str == actual_str:
            return True, True, False
        
        # For numeric values, check if they're close
        try:
            pred_num = float(predicted)
            actual_num = float(actual)
            if abs(pred_num - actual_num) < 0.01:  # Within 1 cent for currency
                return True, True, False
            elif abs(pred_num - actual_num) / max(actual_num, 0.01) < 0.05:  # Within 5%
                return True, False, True
        except (ValueError, TypeError):
            pass
        
        # For strings, check partial matches
        if len(pred_str) > 3 and len(actual_str) > 3:
            if pred_str in actual_str or actual_str in pred_str:
                return True, False, True
            
            # Check for similar strings (simple approach)
            common_chars = len(set(pred_str) & set(actual_str))
            total_chars = len(set(pred_str) | set(actual_str))
            if total_chars > 0 and common_chars / total_chars > 0.7:
                return True, False, True
        
        return False, False, False
    
    def save_to_file(self, file_path: str):
        """Save dataset to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, default=str, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'EvaluationDataset':
        """Load dataset from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.model_validate(data)


# Field importance mapping for waste management invoices
FIELD_IMPORTANCE_MAPPING = {
    "invoice_number": FieldImportance.CRITICAL,
    "current_invoice_charges": FieldImportance.CRITICAL,
    "customer_id": FieldImportance.HIGH,
    "customer_name": FieldImportance.HIGH,
    "invoice_date": FieldImportance.HIGH,
    "service_period": FieldImportance.MEDIUM,
    "line_items": FieldImportance.MEDIUM,
    "vendor_name": FieldImportance.MEDIUM,
    "vendor_address": FieldImportance.LOW,
    "vendor_phone": FieldImportance.LOW,
    "remit_to_address": FieldImportance.LOW,
    "service_location_address": FieldImportance.MEDIUM,
    "gl_account_code": FieldImportance.HIGH,
    "tax_code": FieldImportance.HIGH,
    "from_email_address": FieldImportance.LOW,
    "to_email_address": FieldImportance.LOW,
    "email_date": FieldImportance.LOW,
    "email_subject": FieldImportance.LOW,
    "email_body_content": FieldImportance.LOW,
}