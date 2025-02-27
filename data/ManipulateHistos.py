import sys
import os
from ROOT import TH1, TFile, TObject
from array import *
import math


#########################################################################
#
#  For manipulating histograms
#
#   - eFEXEleStudies: Recalculate the efficiency histograms
#
#########################################################################
class ManipulateHistos(object):

    def __init__(self):
        """ """

    def fixL1CaloJetEffHistos(self, fileStrList):
        """
        L1CaloJetEfficiency Analysis
        Take file list, open file, change histos, close file
        """
        print(fileStrList)
        for fileStr in fileStrList:
            rootFile = TFile(fileStr, "UPDATE")
            self.rootFile = rootFile
            print("here")
            self.fixL1CaloJetEffHistosForFile()
            rootFile.Close()

    def fixEFEXEffHistos(self, fileStrList):
        """
        Take file list, open file, change histos, close file
        """
        for fileStr in fileStrList:
            rootFile = TFile(fileStr, "UPDATE")
            self.rootFile = rootFile
            self.fixEFEXEffHistosForFile()
            rootFile.Close()

    def fixEFEXTauEffHistos(self, fileStrList, prefix=""):
        """
        Take file list, open file, change histos, close file
        """
        for fileStr in fileStrList:
            rootFile = TFile(fileStr, "UPDATE")
            self.rootFile = rootFile
            self.fixEFEXTauEffHistosForFile(prefix=prefix)
            rootFile.Close()

    def fixL1CaloJetEffHistosForFile(self):
        """
        update the L1CaloJetEfficiency histograms - ignore TH2F (map in the name for now)
        """
        objectNames = [key.GetName() for key in self.rootFile.GetListOfKeys()]
        efficiency_prefix = f"h_efficiency_"
        num_postfix = f"_num"
        den_postfix = f"_den"
        histnames = [
            objName
            for objName in objectNames
            if (
                objName.startswith(efficiency_prefix)
                and num_postfix not in objName
                and den_postfix not in objName
                and "_map" not in objName
            )
        ]
        # self.rootFile.ls()
        print(histnames)

        for histname in histnames:
            eff_name = histname
            num_name = eff_name + num_postfix
            denom_name = eff_name + den_postfix
            print("eff_name", eff_name, "num_name", num_name, "denom_name", denom_name)
            self.fixeffHist(eff_name, num_name, denom_name)

    def fixEFEXTauEffHistosForFile(self, prefix=""):
        objectNames = [key.GetName() for key in self.rootFile.GetListOfKeys()]
        efficiency_prefix = f"h_{prefix}efficiency_"
        numerator_prefix = f"h_{prefix}numerator_"
        denominator = f"h_{prefix}denominator_tau"
        histnames = [
            objName[len(efficiency_prefix) :]
            for objName in objectNames
            if objName.startswith(efficiency_prefix)
        ]
        print(histnames)
        for histname in histnames:
            self.fixeffHist(
                efficiency_prefix + histname, numerator_prefix + histname, denominator
            )

    def fixEFEXEffHistosForFileDev(self):
        objectNames = [key.GetName() for key in self.rootFile.GetListOfKeys()]
        efficiency_prefix = f"h_efficiency"
        numerator_prefix = f"h_numerator"
        denominator = f"h_denominator_tau"
        histnames = [
            objName[len(efficiency_prefix) :]
            for objName in objectNames
            if objName.startswith(efficiency_prefix)
        ]
        print(histnames)
        for histname in histnames:
            self.fixeffHist(histname, numerator_prefix + histname, denominator)

    def fixEFEXEffHistosForFile(self):
        """
        Histograms based on eFEXEleStudies :: finalizeEffHists
        """
        # self.rootFile.ls()
        print("Fixing efficiency histos for:", self.rootFile.GetName())

        # https://root.cern/doc/v608/rebin_8C_source.html
        # hdenom = self.rootFile.Get("h_denominator_PhaseI")
        # for bin in range(hdenom.GetNbinsX()+1):
        #    if hdenom.GetBinLowEdge(bin) >= 40.0:
        #        print("bigger than 40",hdenom.GetBinLowEdge(bin),hdenom.GetBinContent(bin))
        # return

        self.fixeffHist(
            "h_efficiency0_PhaseI", "h_numerator0_PhaseI", "h_denominator_PhaseI"
        )
        self.fixeffHist(
            "h_efficiency10_PhaseI", "h_numerator10_PhaseI", "h_denominator_PhaseI"
        )
        self.fixeffHist(
            "h_efficiency15_PhaseI", "h_numerator15_PhaseI", "h_denominator_PhaseI"
        )
        self.fixeffHist(
            "h_efficiency25_PhaseI", "h_numerator25_PhaseI", "h_denominator_PhaseI"
        )
        self.fixeffHist(
            "h_efficiency30_PhaseI", "h_numerator30_PhaseI", "h_denominator_PhaseI"
        )

        self.fixeffHist(
            "h_efficiency18_IB_PhaseI",
            "h_numerator18_IB_PhaseI",
            "h_denominator_IB_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency18_OB_PhaseI",
            "h_numerator18_OB_PhaseI",
            "h_denominator_OB_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency18_OVL_PhaseI",
            "h_numerator18_OVL_PhaseI",
            "h_denominator_OVL_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency18_IEC_PhaseI",
            "h_numerator18_IEC_PhaseI",
            "h_denominator_IEC_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency18_OEC_PhaseI",
            "h_numerator18_OEC_PhaseI",
            "h_denominator_OEC_PhaseI",
        )

        self.fixeffHist(
            "h_efficiency22_noOVL_PhaseI",
            "h_numerator22_noOVL_PhaseI",
            "h_denominator_noOVL_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_IB_PhaseI",
            "h_numerator22_IB_PhaseI",
            "h_denominator_IB_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_OB_PhaseI",
            "h_numerator22_OB_PhaseI",
            "h_denominator_OB_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_OVL_PhaseI",
            "h_numerator22_OVL_PhaseI",
            "h_denominator_OVL_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_IEC_PhaseI",
            "h_numerator22_IEC_PhaseI",
            "h_denominator_IEC_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_OEC_PhaseI",
            "h_numerator22_OEC_PhaseI",
            "h_denominator_OEC_PhaseI",
        )

        self.fixeffHist(
            "h_efficiency22eta_PhaseI",
            "h_numerator22eta_PhaseI",
            "h_denominator22eta_PhaseI",
        )

        # inclusive
        self.fixeffHist(
            "h_efficiency22_PhaseI", "h_numerator22_PhaseI", "h_denominator_PhaseI"
        )
        self.fixeffHist(
            "h_efficiency22_noOVL_PhaseI",
            "h_numerator22_noOVL_PhaseI",
            "h_denominator_noOVL_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_IB_PhaseI",
            "h_numerator22_IB_PhaseI",
            "h_denominator_IB_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_OB_PhaseI",
            "h_numerator22_OB_PhaseI",
            "h_denominator_OB_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_OVL_PhaseI",
            "h_numerator22_OVL_PhaseI",
            "h_denominator_OVL_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_IEC_PhaseI",
            "h_numerator22_IEC_PhaseI",
            "h_denominator_IEC_PhaseI",
        )
        self.fixeffHist(
            "h_efficiency22_OEC_PhaseI",
            "h_numerator22_OEC_PhaseI",
            "h_denominator_OEC_PhaseI",
        )

        # menu
        fullMenuList = [
            "",
            "_2023c",
            "_oneps",
            "_rhadv2",
            "_rhadv2_oneps",
            "_dmcfit2024",
        ]

        #
        menuList = []
        for menutest in fullMenuList:
            if self.rootFile.Get("h_efficiency22_PhaseI_L" + menutest):
                menuList.append(menutest)
                print("found menu:", menutest)

        # HLT
        HLTmap = ["HLT_e26", "HLT_e60"]
        for hlttag in HLTmap:
            self.fixeffHist(
                "h_efficiency22_" + hlttag + "_PhaseI",
                "h_numerator22_" + hlttag + "_PhaseI",
                "h_denominator_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_noOVL_" + hlttag + "_PhaseI",
                "h_numerator22_noOVL_" + hlttag + "_PhaseI",
                "h_denominator_noOVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IB_" + hlttag + "_PhaseI",
                "h_numerator22_IB_" + hlttag + "_PhaseI",
                "h_denominator_IB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OB_" + hlttag + "_PhaseI",
                "h_numerator22_OB_" + hlttag + "_PhaseI",
                "h_denominator_OB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OVL_" + hlttag + "_PhaseI",
                "h_numerator22_OVL_" + hlttag + "_PhaseI",
                "h_denominator_OVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IEC_" + hlttag + "_PhaseI",
                "h_numerator22_IEC_" + hlttag + "_PhaseI",
                "h_denominator_IEC_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OEC_" + hlttag + "_PhaseI",
                "h_numerator22_OEC_" + hlttag + "_PhaseI",
                "h_denominator_OEC_PhaseI",
            )

        for hlt, hlttag in enumerate(HLTmap):
            for l1 in range(2):
                for menu in menuList:
                    l1tag = "_M" if l1 == 0 else "_T"
                    l1tag += menu
                    self.fixeffHist(
                        "h_efficiency22_" + hlttag + "_PhaseI" + l1tag,
                        "h_numerator22_" + hlttag + "_PhaseI" + l1tag,
                        "h_denominator_PhaseI",
                    )
                    self.fixeffHist(
                        "h_efficiency22_noOVL_" + hlttag + "_PhaseI" + l1tag,
                        "h_numerator22_noOVL_" + hlttag + "_PhaseI" + l1tag,
                        "h_denominator_noOVL_PhaseI",
                    )
                    self.fixeffHist(
                        "h_efficiency22_IB_" + hlttag + "_PhaseI" + l1tag,
                        "h_numerator22_IB_" + hlttag + "_PhaseI" + l1tag,
                        "h_denominator_IB_PhaseI",
                    )
                    self.fixeffHist(
                        "h_efficiency22_OB_" + hlttag + "_PhaseI" + l1tag,
                        "h_numerator22_OB_" + hlttag + "_PhaseI" + l1tag,
                        "h_denominator_OB_PhaseI",
                    )
                    self.fixeffHist(
                        "h_efficiency22_OVL_" + hlttag + "_PhaseI" + l1tag,
                        "h_numerator22_OVL_" + hlttag + "_PhaseI" + l1tag,
                        "h_denominator_OVL_PhaseI",
                    )
                    self.fixeffHist(
                        "h_efficiency22_IEC_" + hlttag + "_PhaseI" + l1tag,
                        "h_numerator22_IEC_" + hlttag + "_PhaseI" + l1tag,
                        "h_denominator_IEC_PhaseI",
                    )
                    self.fixeffHist(
                        "h_efficiency22_OEC_" + hlttag + "_PhaseI" + l1tag,
                        "h_numerator22_OEC_" + hlttag + "_PhaseI" + l1tag,
                        "h_denominator_OEC_PhaseI",
                    )

        for menu in menuList:
            self.fixeffHist(
                "h_efficiency22_PhaseI_L" + menu,
                "h_numerator22_PhaseI_L" + menu,
                "h_denominator_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_PhaseI_M" + menu,
                "h_numerator22_PhaseI_M" + menu,
                "h_denominator_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_PhaseI_T" + menu,
                "h_numerator22_PhaseI_T" + menu,
                "h_denominator_PhaseI",
            )
            # fix me
            self.fixeffHist(
                "h_efficiency22_noOVL_PhaseI_L" + menu,
                "h_numerator22_noOVL_PhaseI_L" + menu,
                "h_denominator_noOVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_noOVL_PhaseI_M" + menu,
                "h_numerator22_noOVL_PhaseI_M" + menu,
                "h_denominator_noOVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_noOVL_PhaseI_T" + menu,
                "h_numerator22_noOVL_PhaseI_T" + menu,
                "h_denominator_noOVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IB_PhaseI_L" + menu,
                "h_numerator22_IB_PhaseI_L" + menu,
                "h_denominator_IB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OB_PhaseI_L" + menu,
                "h_numerator22_OB_PhaseI_L" + menu,
                "h_denominator_OB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OVL_PhaseI_L" + menu,
                "h_numerator22_OVL_PhaseI_L" + menu,
                "h_denominator_OVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IEC_PhaseI_L" + menu,
                "h_numerator22_IEC_PhaseI_L" + menu,
                "h_denominator_IEC_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OEC_PhaseI_L" + menu,
                "h_numerator22_OEC_PhaseI_L" + menu,
                "h_denominator_OEC_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IB_PhaseI_M" + menu,
                "h_numerator22_IB_PhaseI_M" + menu,
                "h_denominator_IB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OB_PhaseI_M" + menu,
                "h_numerator22_OB_PhaseI_M" + menu,
                "h_denominator_OB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OVL_PhaseI_M" + menu,
                "h_numerator22_OVL_PhaseI_M" + menu,
                "h_denominator_OVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IEC_PhaseI_M" + menu,
                "h_numerator22_IEC_PhaseI_M" + menu,
                "h_denominator_IEC_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OEC_PhaseI_M" + menu,
                "h_numerator22_OEC_PhaseI_M" + menu,
                "h_denominator_OEC_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IB_PhaseI_T" + menu,
                "h_numerator22_IB_PhaseI_T" + menu,
                "h_denominator_IB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OB_PhaseI_T" + menu,
                "h_numerator22_OB_PhaseI_T" + menu,
                "h_denominator_OB_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OVL_PhaseI_T" + menu,
                "h_numerator22_OVL_PhaseI_T" + menu,
                "h_denominator_OVL_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_IEC_PhaseI_T" + menu,
                "h_numerator22_IEC_PhaseI_T" + menu,
                "h_denominator_IEC_PhaseI",
            )
            self.fixeffHist(
                "h_efficiency22_OEC_PhaseI_T" + menu,
                "h_numerator22_OEC_PhaseI_T" + menu,
                "h_denominator_OEC_PhaseI",
            )

        #
        doEtaPrint = True
        if doEtaPrint:
            for i in range(25):
                etaval = str(i)
                hname = "h_EM_Reta_" + etaval
                hreta = self.rootFile.Get(hname)
                print("Reta ieta:", etaval, hreta.GetEntries())

        # legacy
        self.fixeffHist("h_efficiency22", "h_numerator22", "h_denominator")
        self.fixeffHist("h_efficiency22_IB", "h_numerator22_IB", "h_denominator_IB")
        self.fixeffHist("h_efficiency22_OB", "h_numerator22_OB", "h_denominator_OB")
        self.fixeffHist("h_efficiency22_OVL", "h_numerator22_OVL", "h_denominator_OVL")
        self.fixeffHist("h_efficiency22_IEC", "h_numerator22_IEC", "h_denominator_IEC")
        self.fixeffHist("h_efficiency22_OEC", "h_numerator22_OEC", "h_denominator_OEC")

        self.fixeffHist("h_efficiency22_HI", "h_numerator22_HI", "h_denominator")
        self.fixeffHist(
            "h_efficiency22_IB_HI", "h_numerator22_IB_HI", "h_denominator_IB"
        )
        self.fixeffHist(
            "h_efficiency22_OB_HI", "h_numerator22_OB_HI", "h_denominator_OB"
        )
        self.fixeffHist(
            "h_efficiency22_OVL_HI", "h_numerator22_OVL_HI", "h_denominator_OVL"
        )
        self.fixeffHist(
            "h_efficiency22_noOVL_HI", "h_numerator22_noOVL_HI", "h_denominator_noOVL"
        )
        self.fixeffHist(
            "h_efficiency22_IEC_HI", "h_numerator22_IEC_HI", "h_denominator_IEC"
        )
        self.fixeffHist(
            "h_efficiency22_OEC_HI", "h_numerator22_OEC_HI", "h_denominator_OEC"
        )


    def fixeffHist(self, eff, num, denom, effCut=None):
        """
        recalculate efficiency histogram from num and denom (eFEX)
        """
        self.rootFile.cd()
        heff = self.rootFile.Get(eff)
        hnum = self.rootFile.Get(num)
        hdenom = self.rootFile.Get(denom)

        name = heff.GetName()

        heff.Divide(hnum, hdenom, 1.0, 1.0, "B")

        # print out efficiency above cut
        doPrint = effCut
        if doPrint:
            num = 0.0
            denom = 0.0
            for bin in range(hnum.GetNbinsX() + 2):  # include overflow in efficiency
                if hnum.GetBinLowEdge(bin) >= effCut:
                    num += hnum.GetBinContent(bin)
                    denom += hdenom.GetBinContent(bin)
            eff = num / denom if denom > 0 else -1
            print(
                "eff > " + str(effCut) + ":", heff.GetName(), num, denom, round(eff, 3)
            )

        heff.Write(name, TObject.kOverwrite)

    def fixJFEXEffHistos(self, fileStrList):
        for fileStr in fileStrList:
            rootFile = TFile(fileStr, "UPDATE")
            self.rootFile = rootFile
            self.fixJFEXEffHistosForFile()
            rootFile.Close()

    def fixJFEXEffHistos_Met(self, fileStrList):
        for fileStr in fileStrList:
            rootFile = TFile(fileStr, "UPDATE")
            self.rootFile = rootFile
            self.fixJFEXEffHistosForFile_Met()
            rootFile.Close()

    def fixJFEXEffHistosForFile_Met(self):
        """
        Histograms based on jFEXMet :: finalizeEffHists
        """
        # self.rootFile.ls()
        print("Fixing efficiency histos for:", self.rootFile.GetName())

        # legacy XE50
        self.fixeffHist(
            "h_efficiency_legacy_met_50000",
            "h_triggered_legacy_met_50000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_legacy_met_50000_fine",
            "h_triggered_legacy_met_50000_fine",
            "h_reference_dimuon_pt_fine",
        )

        # rate matched jFEX
        self.fixeffHist(
            "h_efficiency_MetAlgo_ncQuantile_0.1_51000",
            "h_triggered_MetAlgo_ncQuantile_0.1_51000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_ncQuantile_0.1_51000_fine",
            "h_triggered_MetAlgo_ncQuantile_0.1_51000_fine",
            "h_reference_dimuon_pt_fine",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_PileUpCorrection_Simulated_49000",
            "h_triggered_MetAlgo_PileUpCorrection_Simulated_49000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_PileUpCorrection_Simulated_49000_fine",
            "h_triggered_MetAlgo_PileUpCorrection_Simulated_49000_fine",
            "h_reference_dimuon_pt_fine",
        )
        self.fixeffHist(
            "h_efficiency_jfexData_met_48000",
            "h_triggered_jfexData_met_48000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_jfexData_met_48000_fine",
            "h_triggered_jfexData_met_48000_fine",
            "h_reference_dimuon_pt_fine",
        )

        # legacy XE35
        self.fixeffHist(
            "h_efficiency_legacy_met_35000",
            "h_triggered_legacy_met_35000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_legacy_met_35000_fine",
            "h_triggered_legacy_met_35000_fine",
            "h_reference_dimuon_pt_fine",
        )

        # rate matched jFEX
        self.fixeffHist(
            "h_efficiency_MetAlgo_ncQuantile_0.1_34000",
            "h_triggered_MetAlgo_ncQuantile_0.1_34000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_ncQuantile_0.1_34000_fine",
            "h_triggered_MetAlgo_ncQuantile_0.1_34000_fine",
            "h_reference_dimuon_pt_fine",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_PileUpCorrection_Simulated_35000",
            "h_triggered_MetAlgo_PileUpCorrection_Simulated_35000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_PileUpCorrection_Simulated_35000_fine",
            "h_triggered_MetAlgo_PileUpCorrection_Simulated_35000_fine",
            "h_reference_dimuon_pt_fine",
        )
        self.fixeffHist(
            "h_efficiency_jfexData_met_34000",
            "h_triggered_jfexData_met_34000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_jfexData_met_34000_fine",
            "h_triggered_jfexData_met_34000_fine",
            "h_reference_dimuon_pt_fine",
        )

        # legacy XE55
        self.fixeffHist(
            "h_efficiency_legacy_met_55000",
            "h_triggered_legacy_met_55000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_legacy_met_55000_fine",
            "h_triggered_legacy_met_55000_fine",
            "h_reference_dimuon_pt_fine",
        )

        # rate matched jFEX
        self.fixeffHist(
            "h_efficiency_MetAlgo_ncQuantile_0.1_57000",
            "h_triggered_MetAlgo_ncQuantile_0.1_57000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_ncQuantile_0.1_57000_fine",
            "h_triggered_MetAlgo_ncQuantile_0.1_57000_fine",
            "h_reference_dimuon_pt_fine",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_PileUpCorrection_Simulated_53000",
            "h_triggered_MetAlgo_PileUpCorrection_Simulated_53000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_MetAlgo_PileUpCorrection_Simulated_53000_fine",
            "h_triggered_MetAlgo_PileUpCorrection_Simulated_53000_fine",
            "h_reference_dimuon_pt_fine",
        )
        self.fixeffHist(
            "h_efficiency_jfexData_met_52000",
            "h_triggered_jfexData_met_52000",
            "h_reference_dimuon_pt",
        )
        self.fixeffHist(
            "h_efficiency_jfexData_met_52000_fine",
            "h_triggered_jfexData_met_52000_fine",
            "h_reference_dimuon_pt_fine",
        )

    def fixJFEXEffHistosForFile(self):
        objectNames = [key.GetName() for key in self.rootFile.GetListOfKeys()]
        efficiencyNames = [
            objName for objName in objectNames if objName[:12] == "h_efficiency"
        ]
        print(efficiencyNames)
        for effName in efficiencyNames:
            self.fixjFEXeffHist(effName)

    def fixjFEXeffHist(self, effName):
        eff = self.rootFile.Get(effName)

        suffix = effName[12:]  # everything after h_efficiency
        if "_by_eta" in suffix:
            numname = "h_triggered_jet" + suffix[3:]
            denomname = "h_reference_jet" + suffix[3:]
        elif "_forward" in suffix:
            numname = "h_triggered_forward_jet_pt" + suffix[8:]
            denomname = "h_reference_forward_jet_pt" + suffix[8:]
        else:
            numname = "h_triggered_jet_pt" + suffix
            denomname = "h_reference_jet_pt" + suffix

        num = self.rootFile.Get(numname)
        denom = self.rootFile.Get(denomname)

        print("recalculating", effName, "=", numname, "/", denomname)

        eff.Divide(num, denom, "b(1,1) mode")

        eff.Write(effName, TObject.kOverwrite)

    def fixMultiJetHistos(self, fileStrList):
        for fileStr in fileStrList:
            rootFile = TFile(fileStr, "UPDATE")
            self.rootFile = rootFile
            self.fixMultiJetHistosForFile()
            rootFile.Close()

    def fixMultiJetHistosForFile(self):
        objectNames = [key.GetName() for key in self.rootFile.GetListOfKeys()]
        efficiencyNames = [
            objName for objName in objectNames if objName[:12] == "h_efficiency"
        ]
        print(efficiencyNames)
        for effName in efficiencyNames:
            self.fixMultiJetHist(effName)

    def fixMultiJetHist(self, effName):
        eff = self.rootFile.Get(effName)

        suffix = effName[12:]  # everything after h_efficiency

        DRname = ""
        if "closeby" in suffix:
            DRname = "_closeby"
        if "isolated" in suffix:
            DRname = "_isolated"

        suffix = "".join(suffix.split("_closeby"))
        suffix = "".join(suffix.split("_isolated"))

        if "second" in suffix:
            if "jet" in suffix:
                numname = "h_triggered_jet_second_pt" + DRname + suffix[11:]
                denomname = "h_reference_jet_second_pt" + DRname + suffix[11:]
            else:
                numname = "h_triggered_jet" + DRname + "_second_pt" + suffix[7:]
                denomname = "h_reference_jet" + DRname + "_second_pt" + suffix[7:]
        elif "third" in suffix:
            if "jet" in suffix:
                numname = "h_triggered_jet_third_pt" + DRname + suffix[10:]
                denomname = "h_reference_jet_third_pt" + DRname + suffix[10:]
            else:
                numname = "h_triggered_jet" + DRname + "_third_pt" + suffix[6:]
                denomname = "h_reference_jet" + DRname + "_third_pt" + suffix[6:]
        elif "fourth" in suffix:
            if "jet" in suffix:
                numname = "h_triggered_jet_fourth_pt" + DRname + suffix[11:]
                denomname = "h_reference_jet_fourth_pt" + DRname + suffix[11:]
            else:
                numname = "h_triggered_jet" + DRname + "_fourth_pt" + suffix[7:]
                denomname = "h_reference_jet" + DRname + "_fourth_pt" + suffix[7:]
        elif "fifth" in suffix:
            if "jet" in suffix:
                numname = "h_triggered_jet_fifth_pt" + DRname + suffix[10:]
                denomname = "h_reference_jet_fifth_pt" + DRname + suffix[10:]
            else:
                numname = "h_triggered_jet_fifth_pt" + suffix[6:]
                denomname = "h_reference_jet_fifth_pt" + suffix[6:]

        num = self.rootFile.Get(numname)
        denom = self.rootFile.Get(denomname)

        print("recalculating", effName, "=", numname, "/", denomname)

        eff.Divide(num, denom, "b(1,1) mode")

        eff.Write(effName, TObject.kOverwrite)

    def fixJFEXDuplicateJetHistos(self, fileStrList):
        for fileStr in fileStrList:
            rootFile = TFile(fileStr, "UPDATE")
            self.rootFile = rootFile
            self.fixJFEXDuplicateJetHistosForFile()
            rootFile.Close()

    def fixJFEXDuplicateJetHistosForFile(self):
        objectNames = [key.GetName() for key in self.rootFile.GetListOfKeys()]
        efficiencyNames = [
            objName for objName in objectNames if "probability" in objName
        ]
        print(efficiencyNames)
        for effName in efficiencyNames:
            self.fixJFEXDuplicateJetHist(effName)

    def fixJFEXDuplicateJetHist(self, effName):
        eff = self.rootFile.Get(effName)

        numname = "".join(effName.split("_probability"))
        denomname = "matched".join(numname.split("duplicate"))

        num = self.rootFile.Get(numname)
        denom = self.rootFile.Get(denomname)

        print("recalculating", effName, "=", numname, "/", denomname)

        eff.SetTotalHistogram(denom, "f")
        eff.SetPassedHistogram(num, "f")

        eff.Write(effName, TObject.kOverwrite)

    def printRates(self, fileStrList):
        """
        print the trigger rates from a merged eFEXEleRateCount job
        """
        rootFile = TFile(fileStrList[0])
        rootFile.ls("*TrigEvents*")
        
        h_tevts = rootFile.Get("TrigEvents")
        for bin in range(1, 6):
            print(
                f"{h_tevts.GetXaxis().GetBinLabel(bin):<20}:",
                f"{str(int(h_tevts.GetBinContent(bin))):>3}",
            )

        menus = [
            #"VHI",
            #"VHIRemake",
            "",
            "2023c",
            "oneps",
            "rhadv2",
            "rhadv2_oneps",
            "dmcfit2024"
        ]


        # Set up per isolation variable, if available
        varmap = { 0 : "Reta" , 1 : "Rhad", 2 : "Wsto"}
        binmap = { 0 : 8 , 1 : 11, 2 : 14}            
        extraInfo = False # from length of TrigEvents histo
        
        # Print the counts
        print(f'{"WP":20}', f'{" L   /  M  /  T ":10}')
        for menu in menus:
            hist = rootFile.Get("TrigEvents_" + menu) if menu else rootFile.Get("TrigEvents")            
            if "TH1" not in type(hist).__name__:
                continue
            loose = int(hist.GetBinContent(6))
            medium = int(hist.GetBinContent(7))
            tight = int(hist.GetBinContent(8))
            print(
                f"{menu:<20}",
                f"{str(loose):>5} /",
                f"{str(medium):>5} /",
                f"{str(tight):>5}",
            )
            if not extraInfo and hist.GetNbinsX() > 8:
                extraInfo = True

        # and for histograms with extra info
        if extraInfo:
            for i in range(3):
                var = varmap[i]
                print(f'{"WP  (" + var + ")":20}', f'{" L   /  M  /  T ":10}')
                for menu in menus:
                    hist = rootFile.Get("TrigEvents_" + menu) if menu else rootFile.Get("TrigEvents")            
                    if "TH1" not in type(hist).__name__:
                        continue
                    binstart = binmap[i] 
                    loose = int(hist.GetBinContent(binstart + 1))
                    medium = int(hist.GetBinContent(binstart + 2))
                    tight = int(hist.GetBinContent(binstart + 3))
                    print(
                        f"{menu:<20}",
                        f"{str(loose):>5} /",
                        f"{str(medium):>5} /",
                        f"{str(tight):>5}" )

        # Print the rates
        # set the rate according to the run using the filename
        datafile_str = fileStrList[0]

        # alternative approach, if not using eEM18 as reference
        #rate = (num events passing)/(num ZB total) * 40 MHz * Nbunchcoll/Nbunchring
        altCalc = False
        
        
        # eEM18 rate
        rate_kHz = 293
        rateDict = {"486026" : 294, "486179" : 293, "486224" : 293, "486315" : 292}

        present_run = ""
        for run,rate in rateDict.items():
            if run + ".root" in datafile_str:
                rate_kHz = rate
                present_run = run
        # extract run number from filename if not in above list
        if not present_run:
            split_str= datafile_str.split(".root")
            fullrun_str = split_str[0][-8:]
            present_run = fullrun_str[-6:] if fullrun_str.startswith("00") else fullrun_str
                
        print("using eEM18 rate of", rate_kHz, "for run", present_run)

        npresel = h_tevts.GetBinContent(2)
        passPresel = True
        if not npresel>0:
            print("no events passing preselection:", npresel, "run:", present_run)
            passPresel = False
        for menu in menus:
            hist = (
                rootFile.Get("TrigEvents_" + menu) if menu else rootFile.Get("TrigEvents")
            )
            if "TH1" not in type(hist).__name__:
                continue
            loose = round(hist.GetBinContent(6) * rate_kHz / npresel, 1) if passPresel else 0.
            medium = round(hist.GetBinContent(7) * rate_kHz / npresel, 1) if passPresel else 0.
            tight = round(hist.GetBinContent(8) * rate_kHz / npresel, 1) if passPresel else 0.
            if altCalc:
                nbunchcol = 2340
                nbunchring = 3564
                # if we really ran without eEM18 then bin 1 ("total") is the same as ("presel")
                # in a standard job then for alternative estimate then need to use "total"
                #npresel_alt = npresel
                npresel_alt = h_tevts.GetBinContent(1)

                # kHz
                loose = round(hist.GetBinContent(6) * 40e3 * nbunchcol / nbunchring / npresel_alt, 1) 
                medium = round(hist.GetBinContent(7) * 40e3 * nbunchcol/ nbunchring / npresel_alt, 1)
                tight = round(hist.GetBinContent(8) * 40e3 * nbunchcol / nbunchring / npresel_alt, 1)
                
            print(
                f"{menu:<20}",
                f"{str(loose):>5} /",
                f"{str(medium):>5} /",
                f"{str(tight):>5}",
            )

        # and with extra info
        if extraInfo:
            for i in range(3):
                var = varmap[i]
                print(f'{"WP  (" + var + ")":20}', f'{" L   /  M  /  T ":10}')
                for menu in menus:
                    hist = rootFile.Get("TrigEvents_" + menu) if menu else rootFile.Get("TrigEvents")            
                    if "TH1" not in type(hist).__name__:
                        continue
                    binstart = binmap[i] 
                    loose = round(hist.GetBinContent(binstart + 1) * rate_kHz / npresel, 1) if passPresel else 0.
                    medium = round(hist.GetBinContent(binstart + 2) * rate_kHz / npresel, 1) if passPresel else 0.
                    tight = round(hist.GetBinContent(binstart + 3) * rate_kHz / npresel, 1) if passPresel else 0.
                    print(
                        f"{menu:<20}",
                        f"{str(loose):>5} /",
                        f"{str(medium):>5} /",
                        f"{str(tight):>5}")


def main():
    import glob

    # import sys

    import argparse

    parser = argparse.ArgumentParser(
        prog="python ManipulateHistos.py",
        description="""Recalculate trigger efficiencies from source for hadded files\n\n
                   Example: python ManipulateHistos.py --filesInput "data22*" --alg """,
    )
    parser.add_argument(
        "--filesInput", nargs="+", help="input histogram files", required=True
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices={
            "eFEXEleStudies",
            "eFEXEleRateCount",
            "jFEXEfficiency",
            "jFEXMultiJet",
            "jFEXDuplicateJets",
            "eFEXTauStudies",
            "eFEXtauTagProbe",
            "eFEXTauRateCount",
            "jFEXMet",
            "L1CaloJetEfficiency",
        },
        default="Example",
        help="Which analysis to run",
    )
    args = parser.parse_args()

    InputFiles = [file for x in args.filesInput for file in glob.glob(x)]

    print("ManipulateHistos.py main(), analyses:", args.analyses)
    manip = ManipulateHistos()
    if not InputFiles:
        print("Warning: no files found from input list, returning", InputFiles, args.filesInput)
        return

    for alg in args.analyses:
        if "eFEXEleStudies" in alg:
            manip.fixEFEXEffHistos(InputFiles)
        elif "eFEXEleRateCount" in alg:
            manip.printRates(InputFiles)
        elif "jFEXEfficiency" in alg:
            manip.fixJFEXEffHistos(InputFiles)
        elif "jFEXMet" in alg:
            manip.fixJFEXEffHistos_Met(InputFiles)
        elif "jFEXMultiJet" in alg:
            manip.fixMultiJetHistos(InputFiles)
        elif "jFEXDuplicateJets" in alg:
            manip.fixJFEXDuplicateJetHistos(InputFiles)
        elif "eFEXTauStudies" in alg:
            manip.fixEFEXTauEffHistos(InputFiles)
        elif "eFEXtauTagProbe" in alg:
            manip.fixEFEXTauEffHistos(InputFiles, prefix="MC")
        elif "L1CaloJetEfficiency" in alg:
            manip.fixL1CaloJetEffHistos(InputFiles)

    

if __name__ == "__main__":
    main()
