import React, { useState, useEffect } from 'react';
import { supabase } from '../../lib/supabase';
import AgeDemographics from '../AgeDemographics';
import IncomeBracketPie from '../IncomeBracketPie';
import { communityHealthService } from '../../services/communityHealthService';

// Define types for region and patient record
interface Region {
  region_id: string;
  name: string;
}

interface PatientRecord {
  region_id: string;
  body_temperature?: number;
  general_health?: {
    symptom_duration?: string;
  };
  pain_discomfort?: {
    pain_level?: string;
  };
}

interface CommunityData {
  average_age: number;
  total_patients: number;
  patient_names: Array<{
    emotional_state: string;
    sleep_quality: boolean;
    stressful_events: string;
  }>;
}

export const CommunityHealth: React.FC = () => {
  const [regions, setRegions] = useState<Region[]>([]);
  const [selectedRegionId, setSelectedRegionId] = useState<string>('');
  const [patientReports, setPatientReports] = useState<PatientRecord[]>([]);
  const [communityData, setCommunityData] = useState<CommunityData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'demographic' | 'chronic' | 'bmi'>('demographic');

  // Fetch ages directly from the 'appointments' table
  const [ageData, setAgeData] = useState<number[]>([]);
  const [incomeData, setIncomeData] = useState<string[]>([]);

  // Add new state for chronic disease data
  const [chronicDiseaseData, setChronicDiseaseData] = useState<string>('');
  const [chronicLoading, setChronicLoading] = useState(false);
  const [chronicError, setChronicError] = useState<string | null>(null);

  // Fetch regions from Supabase
  useEffect(() => {
    const fetchRegions = async () => {
      const { data, error } = await supabase.from('regions').select('region_id, name');
      if (error) {
        console.error('Error fetching regions:', error.message);
        return;
      }
      setRegions(data || []);
    };

    fetchRegions();
  }, []);

  // Fetch patient health records based on selected region
  useEffect(() => {
    console.log('Selected Region ID:', selectedRegionId); // Debugging log

    const fetchPatientReports = async () => {
      if (!selectedRegionId) {
        console.log('No region selected');
        return;
      }

      const { data, error } = await supabase
        .from('patient_health_records')
        .select('*')
        .eq('region_id', selectedRegionId); // ✅ String-based comparison

      if (error) {
        console.error('Error fetching patient reports:', error.message);
        return;
      }

      console.log('Fetched patient reports:', data); // Debugging log
      setPatientReports(data || []);
    };

    fetchPatientReports();
  }, [selectedRegionId]);

  useEffect(() => {
    const fetchAges = async () => {
      if (!selectedRegionId) return;
      
      console.log('Fetching ages for region:', selectedRegionId);
      
      const { data, error } = await supabase
        .from('appointments')
        .select('age')
        .eq('region_id', selectedRegionId);

      if (error) {
        console.error('Error fetching ages:', error);
        return;
      }

      console.log('Raw data from Supabase:', data);

      if (data) {
        const validAges = data
          .filter((item) => item.age !== null)
          .map((item) => Number(item.age));
          
        console.log('Processed age data:', validAges);
        console.log('Total number of age records:', validAges.length);
        console.log('Age range:', Math.min(...validAges), 'to', Math.max(...validAges));
        
        setAgeData(validAges);
      }
    };

    fetchAges();
  }, [selectedRegionId]);

  useEffect(() => {
    const fetchIncomeData = async () => {
      if (!selectedRegionId) return;
      
      const { data, error } = await supabase
        .from('patient_health_records')
        .select('income_level')
        .eq('region_id', selectedRegionId);

      if (error) {
        console.error('Error fetching income data:', error);
        return;
      }

      if (data) {
        const validIncomes = data
          .filter(item => item.income_level)
          .map(item => item.income_level);
        setIncomeData(validIncomes);
      }
    };

    fetchIncomeData();
  }, [selectedRegionId]);

  // Add new useEffect for fetching chronic disease data
  useEffect(() => {
    const fetchChronicDiseaseData = async () => {
      if (!selectedRegionId || activeTab !== 'chronic') return;
      
      setChronicLoading(true);
      setChronicError(null);
      
      try {
        const data = await communityHealthService.getChronicDiseaseAnalysis(selectedRegionId);
        setChronicDiseaseData(data.analysis);
      } catch (error) {
        setChronicError('Failed to fetch chronic disease analysis');
        console.error('Error:', error);
      } finally {
        setChronicLoading(false);
      }
    };

    fetchChronicDiseaseData();
  }, [selectedRegionId, activeTab]);

  // Update the region selection handler
  const handleRegionChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const regionId = e.target.value;
    setSelectedRegionId(regionId);
    
    if (regionId) {
      setLoading(true);
      setError(null);
      try {
        // const data = await communityHealthService.getCommunityHealth(regionId);
        // console.log('Received from backend:', data);
        // setCommunityData(data);
      } catch (err) {
        setError('Failed to fetch community health data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    } else {
      setCommunityData(null);
    }
  };

  return (
    <div className="bg-gradient-to-br from-gray-50 to-blue-50 min-h-screen p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Community Health Dashboard</h1>
        <select
          value={selectedRegionId}
          onChange={handleRegionChange}
          className="p-2 text-sm border rounded-lg bg-white/90 backdrop-blur-sm shadow-md hover:shadow-lg transition-shadow duration-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-48"
        >
          <option value="">Select Region</option>
          {regions.map((region) => (
            <option key={region.region_id} value={region.region_id}>
              {region.name}
            </option>
          ))}
        </select>
      </div>

      {/* Analysis Tabs */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <button
          onClick={() => setActiveTab('demographic')}
          className={`p-4 text-sm font-medium rounded-t-xl transition-all duration-300 ${
            activeTab === 'demographic'
              ? 'bg-blue-600 text-white shadow-lg transform scale-105'
              : 'bg-white/90 backdrop-blur-sm text-gray-600 hover:bg-blue-50 border border-gray-200 hover:shadow-md'
          }`}
        >
          Demographic & Social
        </button>
        <button
          onClick={() => setActiveTab('chronic')}
          className={`p-4 text-sm font-medium rounded-t-xl transition-all duration-300 ${
            activeTab === 'chronic'
              ? 'bg-green-600 text-white shadow-lg transform scale-105'
              : 'bg-white/90 backdrop-blur-sm text-gray-600 hover:bg-green-50 border border-gray-200 hover:shadow-md'
          }`}
        >
          Chronic Disease
        </button>
        <button
          onClick={() => setActiveTab('bmi')}
          className={`p-4 text-sm font-medium rounded-t-xl transition-all duration-300 ${
            activeTab === 'bmi'
              ? 'bg-purple-600 text-white shadow-lg transform scale-105'
              : 'bg-white/90 backdrop-blur-sm text-gray-600 hover:bg-purple-50 border border-gray-200 hover:shadow-md'
          }`}
        >
          BMI & Vaccination
        </button>
      </div>

      {/* Content Area */}
      <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg p-6 transition-all duration-300 hover:shadow-xl">
        {activeTab === 'demographic' && (
          <div className="space-y-6">
            {/* Age Demographics */}
            <AgeDemographics ages={ageData} />
            
            {/* Income Distribution */}
            <div className="mt-6">
              <IncomeBracketPie incomeData={incomeData} />
            </div>
          </div>
        )}

        {activeTab === 'chronic' && (
          <div className="p-6 space-y-4">
            {chronicLoading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto"></div>
                <p className="mt-4 text-gray-600">Loading analysis...</p>
              </div>
            ) : chronicError ? (
              <div className="text-center py-8 text-red-500">
                <p>{chronicError}</p>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Chronic Disease Analysis</h2>
                <div className="prose max-w-none">
                  {chronicDiseaseData || 'No analysis available for this region'}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'bmi' && (
          <div className="text-center py-8 text-gray-500">
            BMI & vaccination analytics will appear here
          </div>
        )}
      </div>
    </div>
  );
};