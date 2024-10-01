//C, C++
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include <time.h>

using namespace std;

Double_t get_sigma_per_channel( Double_t sigma_ele_ADC, Double_t ADC_per_pe, Double_t Rnsb_Hz, Double_t signalWidth_ns, Double_t measurementsTimeTic_s);

Int_t rate_per_L1_trigger_patch(){
  //
  /*
  TCanvas *c1 = new TCanvas("c1","c1",10,10,600,600);
  gStyle->SetPalette(1);
  gStyle->SetFrameBorderMode(0);
  gROOT->ForceStyle();
  gStyle->SetStatColor(kWhite);
  //gStyle->SetOptStat(kFALSE); 
  */
  //Thesis of David
  // NSB 265 MHz/channel
  //       NSB       Signal
  //0 A    1.6       14.8
  //1 B    1.7       14.9
  //2 C    2.8       16.9
  //3 D    2.2       18.2
  //4 E    2.8       21.4
  //5 F    6.1       22.2
  //6 G    5.6       24.3
  //7 H    8.9       25.7
  //8 I    8.6       30.9
  Double_t sigma_ele_ADC = 4.06;
  Double_t ADC_per_pe = 8.25;
  //Double_t Rnsb_Hz = 265.0*1.0e+6*(8.6/1.6/4.0);
  Double_t Rnsb_Hz = 265.0*1.0e+6;
  Double_t signalWidth_ns = 2.56715;
  Double_t measurementsTimeTic_s = 1000.0/1024.0*1.0e-9;
  //
  Double_t sigma_per_channel = get_sigma_per_channel( sigma_ele_ADC, ADC_per_pe, Rnsb_Hz, signalWidth_ns, measurementsTimeTic_s);
  //Int_t sipm_trg_patch_size = 49;
  Int_t sipm_trg_patch_size = 21;
  //
  Double_t sigma_L1_trigger_patch_SIPM = TMath::Sqrt(sipm_trg_patch_size)*sigma_per_channel;
  Double_t sigma_L1_trigger_patch_SIPM_pe = sigma_L1_trigger_patch_SIPM/ADC_per_pe;
  //
  //
  cout<<"sigma_L1_trigger_patch_SIPM    = "<<sigma_L1_trigger_patch_SIPM<<endl
      <<"sigma_L1_trigger_patch_SIPM_pe = "<<sigma_L1_trigger_patch_SIPM_pe<<endl; 
  //
  //
  //Double_t mazimum_rate_per_L1_trigger_patch = 1.0e+6/1027.0;
  Double_t mazimum_rate_per_L1_trigger_patch = 1.0*1.0e+2;
  Double_t n_sigma_to_trigg = TMath::ErfInverse(1.0 - mazimum_rate_per_L1_trigger_patch*2.0*(1000.0/1024.0*1.0e-9))*TMath::Sqrt(2);
  Double_t n_sigma_to_trigg_pe = n_sigma_to_trigg*sigma_L1_trigger_patch_SIPM_pe;
  //
  //
  cout<<"n_sigma_to_trigg               = "<<n_sigma_to_trigg<<endl
      <<"n_sigma_to_trigg_pe            = "<<n_sigma_to_trigg_pe<<endl;
  //
  //TMath::ErfInverse(1.0 - 2.0*9.5088851e-07);
  //(1.0 - TMath::Erf(5.0/TMath::Sqrt(2)))/2.0/(1000.0/1024.0*1.0e-9)
  //
  //
  /*
  //
  TGraph *gr_80ns_2deg = new TGraph();
  gr_80ns_2deg->SetNameTitle("gr_80ns_2deg","gr_80ns_2deg");
  TGraph *gr_80ns_1deg = new TGraph();
  gr_80ns_1deg->SetNameTitle("gr_80ns_1deg","gr_80ns_1deg");
  TGraph *gr_70ns_07deg = new TGraph();
  gr_70ns_07deg->SetNameTitle("gr_70ns_07deg","gr_70ns_07deg");
  TGraph *gr_80ns_1deg_three = new TGraph();
  gr_80ns_1deg_three->SetNameTitle("gr_80ns_1deg_three","gr_80ns_1deg_three");
  TGraph *gr_70ns_07deg_three = new TGraph();
  gr_70ns_07deg_three->SetNameTitle("gr_70ns_07deg_three","gr_70ns_07deg_three");
  //
  coincident_rate( gr_80ns_2deg,  80*1.0e-9, 2.0, 100000, 3.0*1.0e+6, 1000);
  coincident_rate( gr_80ns_1deg,  80*1.0e-9, 1.0, 100000, 3.0*1.0e+6, 1000);
  coincident_rate( gr_70ns_07deg,  70*1.0e-9, 0.7, 100000, 3.0*1.0e+6, 1000);
  coincident_three_rate(gr_80ns_1deg_three, 80*1.0e-9, 1.0, 100000, 10.0*1.0e+6, 1000);
  coincident_three_rate(gr_70ns_07deg_three, 70*1.0e-9, 0.7, 100000, 10.0*1.0e+6, 1000);
  //
  gr_80ns_1deg->SetLineColor(kBlue);
  gr_80ns_1deg->SetLineWidth(3.0);
  //
  gr_70ns_07deg->SetLineColor(kRed);
  gr_70ns_07deg->SetLineWidth(3.0);
  //
  gr_80ns_1deg_three->SetLineColor(kBlue+2);
  gr_80ns_1deg_three->SetLineWidth(3.0);
  //
  gr_70ns_07deg_three->SetLineColor(kRed+2);
  gr_70ns_07deg_three->SetLineWidth(3.0);
  //
  //gr->Draw("APL");
  //
  TMultiGraph *mg = new TMultiGraph();
  mg->Add(gr_80ns_2deg);
  mg->Add(gr_80ns_1deg);
  //mg->Add(gr_70ns_07deg);
  //mg->Add(gr_80ns_1deg_three);
  //mg->Add(gr_70ns_07deg_three);
  //
  //mg->SetMinimum(1.1e+9);
  mg->SetMaximum(1.0e+8);
  //
  mg->Draw("al");
  //
  gPad->SetLogx();
  gPad->SetLogy();
  gPad->SetGridx();
  gPad->SetGridy();
  //
  mg->GetXaxis()->SetTitle("L0 rate per LST, Hz");
  mg->GetYaxis()->SetTitle("Total fake rate of the 4 LSTs, Hz");
  //
  //
  TLegend *leg = new TLegend(0.6,0.6,0.9,0.9,"","brNDC");
  leg->AddEntry(gr_80ns_2deg,  "#tau = 80 ns no topological trigger", "al");
  leg->AddEntry(gr_80ns_1deg,  "#tau = 80 ns and 1.0 deg topological trigger", "al");
  //leg->AddEntry(gr_70ns_07deg, "#tau = 70 ns and 0.7 deg topological trigger", "al");
  //leg->AddEntry(gr_80ns_1deg_three,  "#tau = 80 ns and 1.0 deg topological trigger (triple coinc.)", "al");
  //leg->AddEntry(gr_70ns_07deg_three, "#tau = 70 ns and 0.7 deg topological trigger (triple coinc.)", "al");
  leg->Draw();
*/
  //
  //
  return 0;
}

Double_t get_sigma_per_channel( Double_t sigma_ele_ADC, Double_t ADC_per_pe, Double_t Rnsb_Hz, Double_t signalWidth_ns, Double_t measurementsTimeTic_s){
  //cout<<(7.855*7.855 - sigma_ele_ADC*sigma_ele_ADC)/(ADC_per_pe*ADC_per_pe*Rnsb_Hz*measurementsTimeTic_s)<<endl;
  return TMath::Sqrt(sigma_ele_ADC*sigma_ele_ADC + ADC_per_pe*ADC_per_pe*Rnsb_Hz*signalWidth_ns*measurementsTimeTic_s);
}
