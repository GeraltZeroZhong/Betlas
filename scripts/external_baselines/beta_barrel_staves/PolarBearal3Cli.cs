using System;
using System.IO;
using betaBarrelProgram;
using betaBarrelProgram.AtomParser;
using betaBarrelProgram.BarrelStructures;
using betaBarrelProgram.Mono;

namespace betaBarrelProgram
{
    public static class ExternalBaselinePolarBearal3Cli
    {
        public static int Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.Error.WriteLine("Usage: PolarBearal3Cli <polarbearal3-source-dir> <input-pdb> <output-dir>");
                return 2;
            }

            string toolDir = Path.GetFullPath(args[0]);
            string inputPdb = Path.GetFullPath(args[1]);
            string outputDir = Path.GetFullPath(args[2]);
            string pdbStem = Path.GetFileNameWithoutExtension(inputPdb);

            Directory.CreateDirectory(outputDir);
            Directory.CreateDirectory(Path.Combine(outputDir, "betaBarrel_aaOnly"));
            Directory.CreateDirectory(Path.Combine(outputDir, "betaBarrelRawData"));
            Directory.CreateDirectory(Path.Combine(outputDir, "betaBarrelStrands"));
            Directory.CreateDirectory(Path.Combine(outputDir, "betaBarrelStrands_UseForCalc"));

            Directory.SetCurrentDirectory(toolDir);
            Global.POLARBEARAL_DIR = toolDir + Path.DirectorySeparatorChar;
            Global.DB_DIR = Path.GetDirectoryName(inputPdb) + Path.DirectorySeparatorChar;
            Global.OUTPUT_DIR = outputDir + Path.DirectorySeparatorChar;
            Global.MONO_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.POLY_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.MEMB_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.SOLUBLE_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.TEST_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.AF_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.AF2_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.AFWeird_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.AF3_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.BIG1_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.CUSTOM_OUTPUT_DIR = Global.OUTPUT_DIR;
            Global.DB_file = Path.Combine(outputDir, "single_pdb_list.txt");
            Global.METHOD = "mono";
            Global.parameterFile = Path.Combine(toolDir, "par_hbond_1.txt");

            File.WriteAllText(Global.DB_file, pdbStem + Environment.NewLine);

            AtomCategory atomCategory = Program.ReadPdbFile(inputPdb, ref Global.partialChargesDict);
            Protein protein = new MonoProtein(ref atomCategory, 0, pdbStem.ToUpper());
            Barrel barrel = new MonoBarrel(protein.Chains[0], protein);

            PolarBearal.PolarBearal_OUTPUT_DIR = Global.OUTPUT_DIR;
            PolarBearal.PolarBearal_INPUT_DB_FILE = Global.DB_file;
            new PolarBearal(ref barrel);

            Console.WriteLine(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}",
                barrel.PdbName,
                barrel.Strands.Count,
                barrel.Axis.Length(),
                barrel.AvgRadius,
                barrel.AvgTilt,
                barrel.ShearNum
            );
            return 0;
        }
    }
}
